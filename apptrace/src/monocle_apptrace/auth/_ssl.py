"""Shared SSL context using certifi's CA bundle.

The python.org macOS Python doesn't populate the system trust store, so
stock urllib HTTPS fails with CERTIFICATE_VERIFY_FAILED on a fresh install.
certifi ships transitively via `requests` (a declared dependency), so it's
reliably present. Fall back to system default if it's not.
"""
import ssl

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()
