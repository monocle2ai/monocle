"""Okahu sign-in via Auth0 (RFC 8252 loopback + RFC 7636 PKCE).

Public entry points called by cli.py:
  - sign_in_via_portal()       loopback flow (browser-based)
  - sign_in_via_device_code()  RFC 8628 device code (no callback URL)
  - sign_in_smart()            auto-pick: loopback on desktop, device when headless
  - resolve_okahu_api_key()    sign in + create/find tenant + mint API key
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Optional

from . import callback_server, okahu_api, pkce, ui
from ._ssl import SSL_CONTEXT
from .config import AuthConfig, load_config


_AUTHORIZE_TIMEOUT_SECONDS = 300
_OAUTH_SCOPES = "openid profile email offline_access"


class PortalAuthError(Exception):
    pass


class LoopbackUnavailable(PortalAuthError):
    """Distinct from PortalAuthError so the smart cascade can fall back to
    the device flow on bind failures without swallowing user-driven errors
    (timeout, cancel, state mismatch)."""


def _is_headless() -> bool:
    if os.environ.get("SSH_TTY") or os.environ.get("SSH_CONNECTION"):
        return True
    if sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return True
    return False


def _build_authorize_url(config: AuthConfig, redirect_uri: str, state: str, code_challenge: str) -> str:
    # No `connection` param — let Auth0's universal login show whichever
    # providers are enabled on the tenant. Pinning to a specific connection
    # breaks if it's not enabled on this client.
    params = {
        "response_type": "code",
        "client_id": config.auth0_client_id,
        "redirect_uri": redirect_uri,
        "scope": _OAUTH_SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "audience": config.auth0_audience,
    }
    return "{}?{}".format(config.authorize_url, urllib.parse.urlencode(params))


def _refresh_access_token(config: AuthConfig, refresh_token: str) -> str:
    # Used after tenant creation: the original token lacks the new tenant_id
    # claim. Refreshing gets a token the API will accept on the mint-key call.
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": config.auth0_client_id,
        "refresh_token": refresh_token,
    }).encode("ascii")
    req = urllib.request.Request(config.token_url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise PortalAuthError("Token refresh failed ({}): {}".format(e.code, detail))
    new_token = data.get("access_token")
    if not new_token:
        raise PortalAuthError("Token refresh response missing access_token.")
    return new_token


def _exchange_code(config: AuthConfig, code: str, code_verifier: str, redirect_uri: str) -> dict:
    body = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "client_id": config.auth0_client_id,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }).encode("ascii")
    req = urllib.request.Request(config.token_url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise PortalAuthError("Token exchange failed ({}): {}".format(e.code, detail))


def _decode_email_from_jwt(access_token: str) -> Optional[str]:
    # No signature verification — Auth0 just issued the token over TLS.
    # We only read the email claim for display purposes.
    try:
        payload_b64 = access_token.split(".")[1]
        padding = "=" * ((4 - len(payload_b64) % 4) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8"))
        return claims.get("email") or claims.get("https://okahu.co/email")
    except (IndexError, ValueError, UnicodeDecodeError):
        return None


def sign_in_via_portal(timeout: float = _AUTHORIZE_TIMEOUT_SECONDS) -> dict:
    """Authorization Code + PKCE flow. Returns {access_token, refresh_token,
    expires_at, email}. Raises PortalAuthError on cancel/timeout/HTTP failure."""
    config = load_config()
    verifier = pkce.generate_code_verifier()
    challenge = pkce.generate_code_challenge(verifier)
    state = pkce.generate_state()

    try:
        port, result, shutdown = callback_server.start(expected_state=state)
    except OSError as e:
        raise LoopbackUnavailable("Cannot bind loopback callback server: {}".format(e))
    # Use "localhost" not "127.0.0.1" — Auth0 does literal-string matching on
    # redirect_uri and the dashboard entry is registered with "localhost".
    redirect_uri = "http://localhost:{}/callback".format(port)
    authorize_url = _build_authorize_url(config, redirect_uri, state, challenge)

    try:
        ui.step("Opening browser to sign in to Okahu...")
        ui.hint("If it doesn't open automatically, visit:")
        ui.hint(authorize_url)
        opened = webbrowser.open(authorize_url)
        if not opened:
            ui.hint("(Could not auto-open a browser — paste the URL above manually.)")

        ui.step("Waiting for sign-in... " + ui.dim("(Ctrl-C to cancel)"))
        try:
            if not result.done.wait(timeout=timeout):
                raise PortalAuthError("Sign-in timed out after {}s.".format(int(timeout)))
        except KeyboardInterrupt:
            raise PortalAuthError("Sign-in cancelled.")

        if result.error:
            raise PortalAuthError(result.error)
        if not result.code:
            raise PortalAuthError("Sign-in completed without an authorization code.")

        token_response = _exchange_code(config, result.code, verifier, redirect_uri)
    finally:
        shutdown()

    access_token = token_response.get("access_token")
    if not access_token:
        raise PortalAuthError("Token endpoint returned no access_token.")

    expires_in = int(token_response.get("expires_in", 0))
    bundle = {
        "access_token": access_token,
        "refresh_token": token_response.get("refresh_token"),
        "expires_at": int(time.time()) + expires_in if expires_in else None,
        "email": _decode_email_from_jwt(access_token),
    }
    if bundle["email"]:
        ui.check("Signed in as " + ui.bold(bundle["email"]))
    return bundle


def _device_code_url(config: AuthConfig) -> str:
    return "https://{}/oauth/device/code".format(config.auth0_domain)


def sign_in_via_device_code(timeout: float = _AUTHORIZE_TIMEOUT_SECONDS) -> dict:
    """Device Authorization Grant (RFC 8628). No callback URL — we poll
    /oauth/token until Auth0 returns tokens. Used when loopback isn't
    available (SSH, headless, port collision)."""
    config = load_config()

    body = urllib.parse.urlencode({
        "client_id": config.auth0_client_id,
        "scope": _OAUTH_SCOPES,
        "audience": config.auth0_audience,
    }).encode("ascii")
    req = urllib.request.Request(_device_code_url(config), data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
            device = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise PortalAuthError("Device-code request failed ({}): {}".format(e.code, detail))

    user_code = device["user_code"]
    verification_uri = device["verification_uri"]
    verification_uri_complete = device.get("verification_uri_complete", verification_uri)
    device_code = device["device_code"]
    interval = int(device.get("interval", 5))
    expires_in = int(device.get("expires_in", 900))

    ui.blank()
    ui.step("To sign in, visit " + ui.brand_alt(verification_uri))
    ui.step("Enter this code:  " + ui.bold(user_code))
    ui.hint("Or open the pre-filled URL: " + verification_uri_complete)
    ui.blank()
    try:
        webbrowser.open(verification_uri_complete)
    except Exception:
        pass
    ui.step("Waiting for sign-in... " + ui.dim("(Ctrl-C to cancel)"))

    deadline = time.time() + min(expires_in, timeout)
    while time.time() < deadline:
        time.sleep(interval)
        poll_body = urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
            "client_id": config.auth0_client_id,
        }).encode("ascii")
        poll_req = urllib.request.Request(config.token_url, data=poll_body, method="POST")
        poll_req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(poll_req, timeout=30, context=SSL_CONTEXT) as resp:
                token_response = json.loads(resp.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as e:
            err_payload = e.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(err_payload).get("error", "")
            except ValueError:
                err = ""
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                interval += 5
                continue
            if err == "expired_token":
                raise PortalAuthError("Device code expired before sign-in completed.")
            if err == "access_denied":
                raise PortalAuthError("Sign-in denied.")
            raise PortalAuthError("Device-flow poll failed ({}): {}".format(e.code, err_payload[:200]))
        except KeyboardInterrupt:
            raise PortalAuthError("Sign-in cancelled.")
    else:
        raise PortalAuthError("Sign-in timed out after {}s.".format(int(min(expires_in, timeout))))

    access_token = token_response.get("access_token")
    if not access_token:
        raise PortalAuthError("Device token endpoint returned no access_token.")

    token_expires_in = int(token_response.get("expires_in", 0))
    bundle = {
        "access_token": access_token,
        "refresh_token": token_response.get("refresh_token"),
        "expires_at": int(time.time()) + token_expires_in if token_expires_in else None,
        "email": _decode_email_from_jwt(access_token),
    }
    if bundle["email"]:
        ui.check("Signed in as " + ui.bold(bundle["email"]))
    return bundle


def sign_in_smart() -> dict:
    """Headless → device flow upfront. Desktop → loopback; if it can't bind,
    fall back to device. User-driven failures (timeout, cancel, state
    mismatch) do not auto-fall-back."""
    if _is_headless():
        ui.step("Headless environment detected — using code sign-in.")
        return sign_in_via_device_code()
    try:
        return sign_in_via_portal()
    except LoopbackUnavailable as e:
        ui.step("{} Falling back to code sign-in.".format(e))
        return sign_in_via_device_code()


def resolve_okahu_api_key(method: str = "smart") -> dict:
    """Sign in → resolve tenant → mint API key.

    `method`: "smart" (default), "loopback", or "device". The returned api_key
    is what the trace exporter sends as `x-api-key`; the access_token is
    used only here during setup.
    """
    if method == "device":
        bundle = sign_in_via_device_code()
    elif method == "loopback":
        bundle = sign_in_via_portal()
    else:
        bundle = sign_in_smart()

    # Cover the silent gap during the two HTTPS calls below (~5s on stage,
    # sub-second on prod).
    ui.step("Setting up your Okahu workspace...")

    config = load_config()
    tenant = okahu_api.fetch_tenant_info(bundle["access_token"], config)
    if tenant is None:
        display = bundle["email"] or "monocle-cli user"
        ui.blank()
        prompt = "No Okahu account found for " + ui.bold(display) + ". Create one?"
        if not ui.confirm(prompt, default_yes=True):
            raise PortalAuthError("Account creation declined.")
        ui.step("Provisioning your Okahu account...")
        tenant = okahu_api.create_tenant(bundle["access_token"], display, config)
        ui.check("Account created")
        # The original token lacks the new tenant_id claim. Refresh so the
        # mint-key call gets a token Auth0 has reissued with the claim.
        if bundle.get("refresh_token"):
            bundle["access_token"] = _refresh_access_token(config, bundle["refresh_token"])

    api_key = okahu_api.mint_api_key(bundle["access_token"], tenant["tenant_id"], config)
    ui.check("API key generated")
    return {
        "api_key": api_key,
        "email": bundle["email"],
        "tenant_id": tenant["tenant_id"],
        "tenant_name": tenant.get("tenant_name"),
    }
