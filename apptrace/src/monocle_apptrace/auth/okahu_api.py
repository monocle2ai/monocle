"""Okahu API calls (Bearer auth): look up tenant, create if missing, mint API key.

The minted API key is what the trace exporter sends as `x-api-key` at
runtime. The access_token is used only here during setup.
"""
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from ._ssl import SSL_CONTEXT
from .config import AuthConfig


# Server-side allow-listed; a value outside the list returns HTTP 400.
# We use VS Code's value because it's known-accepted by the same backend.
# Override via MONOCLE_PROVISIONING_SOURCE if the allow-list adds a CLI tag.
_DEFAULT_PROVISIONING_SOURCE = "Okahu IDE Extension"


class OkahuApiError(Exception):
    def __init__(self, status: int, body: str, message: str = ""):
        super().__init__(message or "Okahu API error {}: {}".format(status, body[:200]))
        self.status = status
        self.body = body


def _pick_first(data: dict, *keys: str) -> Optional[str]:
    # Okahu responses use snake_case, camelCase, or just `id` for the same
    # field across endpoints. Tolerant lookup hides the inconsistency.
    for key in keys:
        value = data.get(key)
        if value:
            return value
    return None


def _bearer_request(url: str, access_token: str, method: str = "GET", body: Optional[dict] = None, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer {}".format(access_token))
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
            payload = resp.read().decode("utf-8")
            return json.loads(payload) if payload.strip() else {}
    except urllib.error.HTTPError as e:
        raise OkahuApiError(e.code, e.read().decode("utf-8", errors="replace"))


def _is_missing_tenant_claim(error: OkahuApiError) -> bool:
    # New users have a token with no tenant_id claim → API returns 401 with
    # this structured error code (NOT a real auth failure — the cue to
    # take the create-tenant branch).
    if error.status != 401:
        return False
    try:
        body = json.loads(error.body)
        return body.get("error", {}).get("code") == "MISSING_TENANT_CLAIM"
    except (ValueError, AttributeError):
        return False


def is_max_api_keys_reached(error: OkahuApiError) -> bool:
    """True iff mint_api_key was rejected because the tenant hit its 8-key cap.

    Caller (cli._get_api_key) routes this to a friendly message that points
    the user at the portal to delete unused keys, rather than dumping the
    raw HTTP error.
    """
    if error.status != 400:
        return False
    body_lower = (error.body or "").lower()
    return "max limit" in body_lower and "api key" in body_lower


def fetch_tenant_info(access_token: str, config: AuthConfig) -> Optional[dict]:
    """Return tenant info if one exists for this user, else None."""
    try:
        data = _bearer_request(config.tenant_url, access_token)
    except OkahuApiError as e:
        if e.status == 404 or _is_missing_tenant_claim(e):
            return None
        raise
    tenant_id = _pick_first(data, "tenant_id", "id", "tenantId")
    if not tenant_id:
        return None
    return {
        "tenant_id": tenant_id,
        "tenant_name": _pick_first(data, "tenant_name", "name", "tenantName"),
    }


def create_tenant(access_token: str, display_name: str, config: AuthConfig) -> dict:
    source = os.environ.get("MONOCLE_PROVISIONING_SOURCE", _DEFAULT_PROVISIONING_SOURCE)
    body = {"display_name": display_name, "provisioning_source": source}
    data = _bearer_request(config.mgmt_tenant_url, access_token, method="POST", body=body)
    tenant_id = _pick_first(data, "tenant_id", "id", "tenantId")
    if not tenant_id:
        raise OkahuApiError(0, json.dumps(data), "Tenant create response missing tenant_id")
    return {
        "tenant_id": tenant_id,
        "tenant_name": _pick_first(data, "tenant_name", "name", "tenantName"),
    }


def mint_api_key(access_token: str, tenant_id: str, config: AuthConfig, key_name: Optional[str] = None) -> str:
    if key_name is None:
        # Names must be unique per tenant (409 CONFLICT on duplicates).
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        key_name = "monocle-cli-{}".format(stamp)
    data = _bearer_request(config.keys_url(tenant_id), access_token, method="POST", body={"name": key_name})
    key = _pick_first(data, "key", "api_key", "apiKey")
    if not key:
        raise OkahuApiError(0, json.dumps(data), "Key mint response missing key field")
    return key
