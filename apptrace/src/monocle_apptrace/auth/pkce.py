"""PKCE (RFC 7636) and CSRF-state helpers for the portal sign-in flow."""
import base64
import hashlib
import secrets


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_code_verifier() -> str:
    # RFC 7636 §4.1: 43-128 chars from [A-Z][a-z][0-9]-._~.
    # base64url(32 random bytes) yields a 43-char verifier in that alphabet.
    return _b64url(secrets.token_bytes(32))


def generate_code_challenge(verifier: str) -> str:
    # RFC 7636 §4.2 S256: BASE64URL(SHA256(ASCII(verifier))).
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def generate_state() -> str:
    return _b64url(secrets.token_bytes(16))
