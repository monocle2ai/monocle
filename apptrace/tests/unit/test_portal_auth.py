"""Unit tests for the auth package. HTTPS calls mocked at urlopen — no network."""
import io
import json
import os
import threading
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


def _fake_http_response(payload, status=200):
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

    def test_create_tenant_posts_display_name(self):
        captured = {}
        def fake_urlopen(req, timeout=None, context=None):
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
        body = b'{"error": {"code": "MISSING_TENANT_CLAIM", "message": "claim missing"}}'
        err = urllib.error.HTTPError(self.config.tenant_url, 401, "Unauthorized",
                                     hdrs={}, fp=io.BytesIO(body))
        with mock.patch.object(urllib.request, "urlopen", side_effect=err):
            self.assertIsNone(okahu_api.fetch_tenant_info(self.access_token, self.config))


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
        import base64
        payload = base64.urlsafe_b64encode(
            json.dumps({"email": "alice@example.com", "sub": "auth0|x"}).encode("utf-8")
        ).rstrip(b"=").decode("ascii")
        fake_jwt = "header.{}.sig".format(payload)
        self.assertEqual(portal_auth._decode_email_from_jwt(fake_jwt), "alice@example.com")


class DeviceFlowTests(unittest.TestCase):
    def _build_jwt_with_email(self, email):
        import base64
        payload = base64.urlsafe_b64encode(
            json.dumps({"email": email, "sub": "auth0|x"}).encode("utf-8")
        ).rstrip(b"=").decode("ascii")
        return "header.{}.sig".format(payload)

    def test_device_polling_handles_authorization_pending_then_success(self):
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
             mock.patch.object(portal_auth.webbrowser, "open"), \
             mock.patch("time.sleep"):
            bundle = portal_auth.sign_in_via_device_code()

        self.assertEqual(call_count["n"], 2)
        self.assertEqual(bundle["email"], "carol@example.com")


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


if __name__ == "__main__":
    unittest.main()
