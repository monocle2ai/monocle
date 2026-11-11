"""Okahu step of `monocle-apptrace <agent>-setup`: get an Okahu API key.

Moved from cli.py unchanged; get_setup_env is the provider hook the CLI calls.
"""
import getpass
import sys

from monocle_apptrace.auth import ui

_DOCS_URL = "https://docs.okahu.ai/"
_VSCODE_EXTENSION_URL = "https://marketplace.visualstudio.com/items?itemName=OkahuAI.okahu-ai-observability"


def get_setup_env(args):
    """Return (env, aborted): the settings to save, or None to use local files."""
    api_key, aborted = _get_api_key(args)
    if not api_key:
        return None, aborted
    return {"OKAHU_API_KEY": api_key, "MONOCLE_EXPORTER": "okahu"}, False


# =============================================================================
# API key resolution
# =============================================================================


def _get_api_key(args):
    """Decide how to get an OKAHU_API_KEY.

    Returns (api_key, aborted):
      api_key is the key string, or None for "no key — use file exporter."
      aborted is True only if the user tried to sign in and it failed.
    """
    from monocle_apptrace.providers.okahu.auth.okahu_api import OkahuApiError, is_max_api_keys_reached
    from monocle_apptrace.providers.okahu.auth.portal_auth import PortalAuthError, resolve_okahu_api_key

    if args.api_key:
        return args.api_key, False
    if args.local_only:
        return None, False

    existing = _read_existing_key()

    # Pick the sign-in mechanism: explicit flag wins; otherwise reuse the
    # existing key (with confirmation if interactive); otherwise prompt.
    if args.portal:
        method = "loopback"
    elif args.device:
        method = "device"
    elif existing and not sys.stdin.isatty():
        return existing, False  # CI / scripts: reuse silently
    elif existing and _confirm_reuse_existing(existing):
        return existing, False
    else:
        method = _prompt_auth_method()

    if method in ("smart", "loopback", "device"):
        try:
            result = resolve_okahu_api_key(method=method)
        except PortalAuthError as e:
            ui.fail("Sign-in failed: " + str(e))
            return None, True
        except OkahuApiError as e:
            if is_max_api_keys_reached(e):
                _explain_max_keys_reached()
            else:
                ui.fail("Okahu API error: " + str(e))
            return None, True
        return result["api_key"], False

    if method == "paste":
        key = getpass.getpass("  Okahu API key: ").strip()
        return (key or None), False

    return None, False  # "skip" or anything else falls through here.


def _read_existing_key():
    # Lazy import: get_monocle_env_value transitively pulls in OpenTelemetry,
    # which we don't want to pay for on `monocle-apptrace --version`.
    from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value
    return get_monocle_env_value("OKAHU_API_KEY")


def _explain_max_keys_reached():
    """User-friendly message for the '8 API keys per tenant' cap."""
    ui.fail("Couldn't generate a new API key — your Okahu tenant has reached the 8-key limit.")
    ui.blank()
    ui.hint("Two ways to recover:")
    ui.hint("  • Reuse one of your existing keys — re-run setup and pick \"Paste an API key\".")
    ui.hint("  • Delete unused keys at https://portal.okahu.co (Settings → API Keys), then re-run.")


def _confirm_reuse_existing(existing_key):
    masked = existing_key[:6] + "…" + existing_key[-4:] if len(existing_key) > 10 else "…"
    if ui.confirm("Continue with existing API key " + ui.dim(masked) + "?", default_yes=True):
        ui.check("Using existing Okahu API key")
        return True
    return False


def _prompt_auth_method():
    if not sys.stdin.isatty():
        return "skip"
    return ui.select(
        "How would you like to authenticate with Okahu?",
        [
            {"key": "smart", "label": "Sign in",
             "hint": "opens your browser, or a code if needed"},
            {"key": "paste", "label": "Paste an API key",
             "hint": "if you already have one from the Okahu portal"},
            {"key": "skip",  "label": "Skip",
             "hint": "store traces locally and inspect with the VS Code Okahu extension"},
        ],
        links=[
            ("learn more", _DOCS_URL),
            ("extension", _VSCODE_EXTENSION_URL),
        ],
    )
