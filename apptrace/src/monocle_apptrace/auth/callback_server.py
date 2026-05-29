"""One-shot loopback HTTP server that receives the OAuth redirect (RFC 8252).

Port is pinned (default 18292) because Auth0 does exact-string matching on
the registered redirect_uri. Override via MONOCLE_LOOPBACK_PORT if needed,
but the Auth0 app's allow-list must also be updated to match.
"""
import html
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse


_DEFAULT_LOOPBACK_PORT = 18292


_PAGE_CSS = (
    "html,body{height:100%;margin:0;}"
    "body{"
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;"
    "background:#0a1820;color:#ffffff;"
    "display:flex;align-items:center;justify-content:center;"
    "text-align:center;-webkit-font-smoothing:antialiased;"
    "}"
    ".wordmark{font-size:36px;font-weight:700;letter-spacing:-0.01em;margin-bottom:28px;}"
    "h1{font-size:22px;font-weight:500;margin:0 0 6px;}"
    "p{color:#6b7d8f;margin:0;font-size:14px;line-height:1.5;}"
)


def _render_page(title: str, message_html: str) -> bytes:
    """Render the callback page. `message_html` is trusted; callers escape input."""
    document = (
        "<!doctype html>"
        "<html lang=\"en\"><head>"
        "<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>{title} · Okahu</title>"
        "<style>{css}</style>"
        "</head><body><div>"
        "<div class=\"wordmark\">Okahu</div>"
        "<h1>{title}</h1>"
        "<p>{message}</p>"
        "</div></body></html>"
    ).format(title=title, css=_PAGE_CSS, message=message_html)
    return document.encode("utf-8")


_SUCCESS_HTML = _render_page("Signed in", "You can close this tab.")


def _error_html(message: str) -> bytes:
    # `message` is Auth0's error_description — escape before injecting.
    return _render_page("Sign-in failed", html.escape(message))


class CallbackResult:
    def __init__(self):
        self.code: Optional[str] = None
        self.error: Optional[str] = None
        self.done = threading.Event()


def _make_handler(expected_state: str, result: CallbackResult):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs):
            # Silence the default per-request stderr line.
            pass

        def _respond(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, message: str) -> None:
            result.error = message
            self._respond(400, _error_html(message))
            result.done.set()

        def do_GET(self):
            params = parse_qs(urlparse(self.path).query)

            err = params.get("error", [None])[0]
            if err:
                self._fail(params.get("error_description", [err])[0])
                return

            if params.get("state", [None])[0] != expected_state:
                self._fail("State mismatch (possible CSRF).")
                return

            code = params.get("code", [None])[0]
            if not code:
                self._fail("No authorization code in callback.")
                return

            result.code = code
            self._respond(200, _SUCCESS_HTML)
            result.done.set()

    return _Handler


def _resolve_port() -> int:
    override = os.environ.get("MONOCLE_LOOPBACK_PORT")
    if override:
        return int(override)
    return _DEFAULT_LOOPBACK_PORT


def start(expected_state: str):
    """Bind a one-shot HTTP server on 127.0.0.1. Returns (port, result,
    shutdown_fn). Raises OSError if the port is already in use — callers
    are expected to catch that and fall back (see portal_auth.sign_in_smart).
    """
    requested_port = _resolve_port()
    result = CallbackResult()
    server = HTTPServer(("127.0.0.1", requested_port), _make_handler(expected_state, result))
    # If requested_port was 0, the OS picked a free port — surface the real one.
    actual_port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def shutdown():
        server.shutdown()
        server.server_close()

    return actual_port, result, shutdown
