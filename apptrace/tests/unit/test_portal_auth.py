"""Unit tests for the portal OAuth sign-in pipeline.

Exercises the auth package end-to-end without touching real Auth0 or Okahu
servers — the loopback callback server is driven by a synthetic browser hit,
and HTTPS calls to Auth0 / Okahu management endpoints are mocked at
urllib.request.urlopen.
"""
import io
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from monocle_apptrace.auth import (
    callback_server,
    config,
    okahu_api,
    portal_auth,
)


class CallbackServerTests(unittest.TestCase):
    def setUp(self):
        # Use an OS-assigned port so tests don't collide on the pinned 18292.
        self._port_patch = mock.patch.dict(os.environ, {"MONOCLE_LOOPBACK_PORT": "0"})
        self._port_patch.start()

    def tearDown(self):
        self._port_patch.stop()

    def _drive(self, query):
        port, result, shutdown = callback_server.start(expected_state="abc")
        try:
            def hit():
                # Tolerate 4xx — server still returns the captured fields.
                try:
                    urllib.request.urlopen(
                        "http://127.0.0.1:{}/callback?{}".format(port, query),
                        timeout=2,
                    ).read()
                except urllib.error.HTTPError:
                    pass
            threading.Thread(target=hit, daemon=True).start()
            self.assertTrue(result.done.wait(timeout=3))
            return result
        finally:
            shutdown()

    def test_captures_code_on_valid_state(self):
        result = self._drive("code=abc123&state=abc")
        self.assertEqual(result.code, "abc123")
        self.assertIsNone(result.error)

    def test_rejects_state_mismatch(self):
        result = self._drive("code=abc123&state=wrong")
        self.assertIsNone(result.code)
        self.assertIn("State mismatch", result.error)

    def test_surfaces_provider_error(self):
        result = self._drive("error=access_denied&error_description=user+denied")
        self.assertIsNone(result.code)
        self.assertIn("user denied", result.error)

    def test_missing_code_param(self):
        result = self._drive("state=abc")
        self.assertIsNone(result.code)
        self.assertIn("No authorization code", result.error)


def _fake_http_response(payload, status=200):
    """Build a context-manager-friendly fake response for urllib.request.urlopen."""
    body = json.dumps(payload).encode("utf-8")
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    resp.status = status
    return resp


class OkahuApiTests(unittest.TestCase):
    def setUp(self):
        self.config = config.load_config()
        self.access_token = "fake-token"

    def test_fetch_tenant_returns_normalized_payload(self):
        with mock.patch.object(urllib.request, "urlopen",
                               return_value=_fake_http_response({"tenant_id": "t1", "tenant_name": "Acme"})):
            tenant = okahu_api.fetch_tenant_info(self.access_token, self.config)
        self.assertEqual(tenant, {"tenant_id": "t1", "tenant_name": "Acme"})

    def test_fetch_tenant_returns_none_on_404(self):
        err = urllib.error.HTTPError(self.config.tenant_url, 404, "Not Found",
                                     hdrs={}, fp=io.BytesIO(b"{}"))
        with mock.patch.object(urllib.request, "urlopen", side_effect=err):
            self.assertIsNone(okahu_api.fetch_tenant_info(self.access_token, self.config))

    def test_create_tenant_posts_display_name(self):
        captured = {}
        def fake_urlopen(req, timeout=None, context=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _fake_http_response({"tenant_id": "t3"})
        with mock.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            okahu_api.create_tenant(self.access_token, "alice@example.com", self.config)
        self.assertEqual(captured["body"]["display_name"], "alice@example.com")
        self.assertEqual(captured["body"]["provisioning_source"], "Okahu IDE Extension")

    def test_mint_api_key_extracts_key_field(self):
        with mock.patch.object(urllib.request, "urlopen",
                               return_value=_fake_http_response({"key": "k-123"})):
            key = okahu_api.mint_api_key(self.access_token, "t1", self.config)
        self.assertEqual(key, "k-123")

    def test_fetch_tenant_returns_none_on_missing_tenant_claim(self):
        # New users: the API responds 401 with a structured body. fetch_tenant_info
        # should treat that as "no tenant" (same as 404), letting the caller
        # take the create-tenant branch.
        body = b'{"error": {"code": "MISSING_TENANT_CLAIM", "message": "claim missing"}}'
        err = urllib.error.HTTPError(self.config.tenant_url, 401, "Unauthorized",
                                     hdrs={}, fp=io.BytesIO(body))
        with mock.patch.object(urllib.request, "urlopen", side_effect=err):
            self.assertIsNone(okahu_api.fetch_tenant_info(self.access_token, self.config))


class ConfigTests(unittest.TestCase):
    def test_env_var_overrides_defaults(self):
        with mock.patch.dict(os.environ, {
            "MONOCLE_AUTH0_DOMAIN": "dev.example.com",
            "MONOCLE_AUTH0_CLIENT_ID": "test-id",
            "MONOCLE_OKAHU_API_HOST": "api.example.com",
        }):
            cfg = config.load_config()
        self.assertEqual(cfg.auth0_domain, "dev.example.com")
        self.assertEqual(cfg.auth0_client_id, "test-id")
        self.assertEqual(cfg.authorize_url, "https://dev.example.com/authorize")
        self.assertEqual(cfg.keys_url("t1"), "https://api.example.com/api/v1/tenants/t1/keys")

    def test_prod_defaults_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in ("MONOCLE_AUTH0_DOMAIN", "MONOCLE_AUTH0_CLIENT_ID",
                        "MONOCLE_AUTH0_AUDIENCE", "MONOCLE_OKAHU_API_HOST",
                        "MONOCLE_OKAHU_MGMT_HOST"):
                os.environ.pop(key, None)
            cfg = config.load_config()
        # Defaults point at the prod Auth0 app (PKCE-only, loopback callback
        # registered) and the prod API hosts that trust its signing keys.
        # Stage is opt-in via env-var overrides.
        self.assertEqual(cfg.auth0_domain, "auth.okahu.co")
        self.assertEqual(cfg.auth0_client_id, "w5msUdJgpximQMqo8LqkrLyiQMYbYiDw")
        self.assertEqual(cfg.okahu_api_host, "api.okahu.co")
        self.assertEqual(cfg.okahu_mgmt_host, "management.okahu.co")


class PortalAuthFlowTests(unittest.TestCase):
    def test_authorize_url_contains_pkce_and_state(self):
        cfg = config.load_config()
        url = portal_auth._build_authorize_url(
            cfg, "http://127.0.0.1:12345/callback", "STATE123", "CHALLENGE123",
        )
        self.assertIn("response_type=code", url)
        self.assertIn("code_challenge=CHALLENGE123", url)
        self.assertIn("code_challenge_method=S256", url)
        self.assertIn("state=STATE123", url)
        self.assertIn("redirect_uri=http%3A%2F%2F127.0.0.1%3A12345%2Fcallback", url)

    def test_jwt_email_extraction(self):
        # Hand-craft a JWT-like string: header.payload.sig — only payload matters.
        import base64
        payload = base64.urlsafe_b64encode(
            json.dumps({"email": "alice@example.com", "sub": "auth0|x"}).encode("utf-8")
        ).rstrip(b"=").decode("ascii")
        fake_jwt = "header.{}.sig".format(payload)
        self.assertEqual(portal_auth._decode_email_from_jwt(fake_jwt), "alice@example.com")


class DeviceFlowTests(unittest.TestCase):
    """Cover the Device Authorization Grant polling state machine."""

    def _build_jwt_with_email(self, email):
        import base64
        payload = base64.urlsafe_b64encode(
            json.dumps({"email": email, "sub": "auth0|x"}).encode("utf-8")
        ).rstrip(b"=").decode("ascii")
        return "header.{}.sig".format(payload)

    def test_device_request_body_carries_client_id_scope_audience(self):
        captured = {"requests": []}

        def fake_urlopen(req, timeout=None, context=None):
            captured["requests"].append((req.full_url, req.data, req.method))
            url = req.full_url
            if "/device/code" in url:
                return _fake_http_response({
                    "device_code": "DEVCODE",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://auth.okahu.co/activate",
                    "verification_uri_complete": "https://auth.okahu.co/activate?user_code=ABCD-EFGH",
                    "expires_in": 600,
                    "interval": 0,
                })
            # Token endpoint: succeed immediately
            return _fake_http_response({
                "access_token": self._build_jwt_with_email("bob@example.com"),
                "expires_in": 3600,
            })

        with mock.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen), \
             mock.patch("time.sleep"):  # interval=0 anyway, but be safe
            bundle = portal_auth.sign_in_via_device_code()

        # First request hit /device/code with the right form-encoded body.
        first_url, first_body, first_method = captured["requests"][0]
        self.assertIn("/oauth/device/code", first_url)
        self.assertEqual(first_method, "POST")
        from urllib.parse import parse_qs
        parsed = parse_qs(first_body.decode("ascii"))
        self.assertIn("client_id", parsed)
        self.assertIn("audience", parsed)
        self.assertIn("openid", parsed["scope"][0])
        self.assertEqual(bundle["email"], "bob@example.com")

    def test_device_polling_handles_authorization_pending_then_success(self):
        # First poll returns 400 authorization_pending, second returns tokens.
        call_count = {"n": 0}

        def fake_urlopen(req, timeout=None, context=None):
            if "/device/code" in req.full_url:
                return _fake_http_response({
                    "device_code": "DC", "user_code": "AB-CD",
                    "verification_uri": "x", "verification_uri_complete": "y",
                    "expires_in": 600, "interval": 0,
                })
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise urllib.error.HTTPError(
                    req.full_url, 400, "Bad", hdrs={},
                    fp=io.BytesIO(b'{"error":"authorization_pending"}'),
                )
            return _fake_http_response({
                "access_token": self._build_jwt_with_email("carol@example.com"),
                "expires_in": 3600,
            })

        with mock.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen), \
             mock.patch("time.sleep"):
            bundle = portal_auth.sign_in_via_device_code()

        self.assertEqual(call_count["n"], 2)  # one pending, one success
        self.assertEqual(bundle["email"], "carol@example.com")

    def test_device_polling_raises_on_access_denied(self):
        def fake_urlopen(req, timeout=None, context=None):
            if "/device/code" in req.full_url:
                return _fake_http_response({
                    "device_code": "DC", "user_code": "AB-CD",
                    "verification_uri": "x", "verification_uri_complete": "y",
                    "expires_in": 600, "interval": 0,
                })
            raise urllib.error.HTTPError(
                req.full_url, 403, "Forbidden", hdrs={},
                fp=io.BytesIO(b'{"error":"access_denied"}'),
            )

        with mock.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen), \
             mock.patch("time.sleep"):
            with self.assertRaises(portal_auth.PortalAuthError) as ctx:
                portal_auth.sign_in_via_device_code()
        self.assertIn("denied", str(ctx.exception).lower())

    def test_device_polling_raises_on_expired_token(self):
        def fake_urlopen(req, timeout=None, context=None):
            if "/device/code" in req.full_url:
                return _fake_http_response({
                    "device_code": "DC", "user_code": "AB-CD",
                    "verification_uri": "x", "verification_uri_complete": "y",
                    "expires_in": 600, "interval": 0,
                })
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad", hdrs={},
                fp=io.BytesIO(b'{"error":"expired_token"}'),
            )

        with mock.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen), \
             mock.patch("time.sleep"):
            with self.assertRaises(portal_auth.PortalAuthError) as ctx:
                portal_auth.sign_in_via_device_code()
        self.assertIn("expired", str(ctx.exception).lower())

    def test_resolve_uses_device_flow_when_method_is_device(self):
        with mock.patch.object(portal_auth, "sign_in_via_device_code") as device, \
             mock.patch.object(portal_auth, "sign_in_via_portal") as loopback, \
             mock.patch.object(portal_auth.okahu_api, "fetch_tenant_info",
                               return_value={"tenant_id": "t", "tenant_name": "T"}), \
             mock.patch.object(portal_auth.okahu_api, "mint_api_key", return_value="k"):
            device.return_value = {"access_token": "tok", "refresh_token": None,
                                   "expires_at": None, "email": "x@y"}
            portal_auth.resolve_okahu_api_key(method="device")
        device.assert_called_once()
        loopback.assert_not_called()


class HeadlessDetectionTests(unittest.TestCase):
    def _clear(self):
        for k in ("SSH_TTY", "SSH_CONNECTION", "DISPLAY", "WAYLAND_DISPLAY"):
            os.environ.pop(k, None)

    def test_ssh_tty_triggers_headless(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            self._clear()
            os.environ["SSH_TTY"] = "/dev/pts/0"
            self.assertTrue(portal_auth._is_headless())

    def test_linux_without_display_is_headless(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(portal_auth.sys, "platform", "linux"):
            self._clear()
            self.assertTrue(portal_auth._is_headless())

    def test_linux_with_display_is_not_headless(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(portal_auth.sys, "platform", "linux"):
            self._clear()
            os.environ["DISPLAY"] = ":0"
            self.assertFalse(portal_auth._is_headless())


class SmartCascadeTests(unittest.TestCase):
    def test_smart_uses_device_when_headless(self):
        with mock.patch.dict(os.environ, {"SSH_TTY": "/dev/pts/0"}, clear=False), \
             mock.patch.object(portal_auth, "sign_in_via_device_code") as device, \
             mock.patch.object(portal_auth, "sign_in_via_portal") as loopback:
            device.return_value = {"access_token": "from-device", "refresh_token": None,
                                   "expires_at": None, "email": "x@y"}
            portal_auth.sign_in_smart()
        device.assert_called_once()
        loopback.assert_not_called()

    def test_smart_uses_loopback_on_desktop(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(portal_auth, "sign_in_via_device_code") as device, \
             mock.patch.object(portal_auth, "sign_in_via_portal") as loopback:
            for k in ("SSH_TTY", "SSH_CONNECTION", "DISPLAY", "WAYLAND_DISPLAY"):
                os.environ.pop(k, None)
            with mock.patch.object(portal_auth.sys, "platform", "darwin"):
                loopback.return_value = {"access_token": "from-loopback", "refresh_token": None,
                                         "expires_at": None, "email": "x@y"}
                portal_auth.sign_in_smart()
        loopback.assert_called_once()
        device.assert_not_called()

    def test_smart_falls_back_to_device_when_loopback_cannot_bind(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(portal_auth, "sign_in_via_device_code") as device, \
             mock.patch.object(portal_auth, "sign_in_via_portal",
                               side_effect=portal_auth.LoopbackUnavailable("port busy")) as loopback:
            for k in ("SSH_TTY", "SSH_CONNECTION", "DISPLAY", "WAYLAND_DISPLAY"):
                os.environ.pop(k, None)
            with mock.patch.object(portal_auth.sys, "platform", "darwin"):
                device.return_value = {"access_token": "from-device", "refresh_token": None,
                                       "expires_at": None, "email": "x@y"}
                portal_auth.sign_in_smart()
        loopback.assert_called_once()
        device.assert_called_once()

    def test_smart_does_not_fall_back_on_user_driven_failure(self):
        # Timeout / cancel / state mismatch must surface, not silently switch flows.
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(portal_auth, "sign_in_via_device_code") as device, \
             mock.patch.object(portal_auth, "sign_in_via_portal",
                               side_effect=portal_auth.PortalAuthError("Sign-in timed out.")):
            for k in ("SSH_TTY", "SSH_CONNECTION", "DISPLAY", "WAYLAND_DISPLAY"):
                os.environ.pop(k, None)
            with mock.patch.object(portal_auth.sys, "platform", "darwin"):
                with self.assertRaises(portal_auth.PortalAuthError):
                    portal_auth.sign_in_smart()
        device.assert_not_called()

    def test_resolve_uses_smart_by_default(self):
        with mock.patch.object(portal_auth, "sign_in_smart") as smart, \
             mock.patch.object(portal_auth.okahu_api, "fetch_tenant_info",
                               return_value={"tenant_id": "t", "tenant_name": "T"}), \
             mock.patch.object(portal_auth.okahu_api, "mint_api_key", return_value="k"):
            smart.return_value = {"access_token": "tok", "refresh_token": None,
                                  "expires_at": None, "email": "x@y"}
            result = portal_auth.resolve_okahu_api_key()  # default method
        smart.assert_called_once()
        self.assertEqual(result["api_key"], "k")

    def test_resolve_with_loopback_method_skips_fallback(self):
        with mock.patch.object(portal_auth, "sign_in_via_portal") as loopback, \
             mock.patch.object(portal_auth, "sign_in_smart") as smart, \
             mock.patch.object(portal_auth.okahu_api, "fetch_tenant_info",
                               return_value={"tenant_id": "t", "tenant_name": "T"}), \
             mock.patch.object(portal_auth.okahu_api, "mint_api_key", return_value="k"):
            loopback.return_value = {"access_token": "tok", "refresh_token": None,
                                     "expires_at": None, "email": "x@y"}
            portal_auth.resolve_okahu_api_key(method="loopback")
        loopback.assert_called_once()
        smart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
