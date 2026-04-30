import json
import shlex
import sys
from pathlib import Path

_MARKER = "monocle_apptrace.instrumentation.metamodel.codex_cli"
_CODEX_DIR = Path.home() / ".codex"
_HOOKS_FILE = _CODEX_DIR / "hooks.json"
_CONFIG_FILE = _CODEX_DIR / "config.toml"
_TEMPLATE = Path(__file__).parent / "hooks.json"


def _pin_interpreter(command: str) -> str:
    """Replace a leading ``python`` token with ``sys.executable``.

    The hook subprocess inherits the user's PATH, which may resolve ``python``
    to a different interpreter than the one that has ``monocle_apptrace``
    installed. Pinning here avoids that silent failure mode.
    """
    parts = shlex.split(command)
    if parts and parts[0] == "python":
        parts[0] = sys.executable
    return shlex.join(parts)


def install() -> int:
    _CODEX_DIR.mkdir(parents=True, exist_ok=True)

    monocle_hooks = json.loads(_TEMPLATE.read_text())["hooks"]
    for groups in monocle_hooks.values():
        for group in groups:
            for hook in group.get("hooks", []):
                hook["command"] = _pin_interpreter(hook.get("command", ""))

    # Merge non-destructively; skip events already carrying our marker.
    try:
        settings = json.loads(_HOOKS_FILE.read_text()) if _HOOKS_FILE.exists() else {}
    except Exception:
        settings = {}
    settings.setdefault("hooks", {})
    added = []
    for event, groups in monocle_hooks.items():
        existing = settings["hooks"].get(event, [])
        if any(_MARKER in h.get("command", "")
               for g in existing for h in g.get("hooks", [])):
            continue
        settings["hooks"].setdefault(event, []).extend(groups)
        added.append(event)
    _HOOKS_FILE.write_text(json.dumps(settings, indent=2))

    # Append-only: don't rewrite TOML structure, just add the flag if missing.
    text = _CONFIG_FILE.read_text() if _CONFIG_FILE.exists() else ""
    flag_added = "codex_hooks = true" not in text
    if flag_added:
        sep = "\n" if text and not text.endswith("\n") else ""
        _CONFIG_FILE.write_text(text + sep + "[features]\ncodex_hooks = true\n")

    print(f"[monocle] Hooks: {', '.join(added) if added else 'already registered'}")
    if flag_added:
        print(f"[monocle] Enabled codex_hooks in {_CONFIG_FILE}")
    print(f"[monocle] Pinned to: {sys.executable}")
    print("[monocle] Start a new Codex session — traces will flow automatically.")
    return 0
