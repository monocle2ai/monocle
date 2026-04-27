import json
import shlex
import sys
from pathlib import Path

_MARKER = "monocle_apptrace.instrumentation.metamodel.claude_cli"
_SETTINGS = Path.home() / ".claude" / "settings.json"
_HOOKS_JSON = Path(__file__).parent / "hooks.json"


def _resolve_command(template: str) -> str:
    """Replace the leading ``python`` token with the interpreter that ran setup.

    Hook processes inherit the user's shell PATH, which may resolve ``python`` to
    a different interpreter than the one with ``monocle_apptrace`` installed.
    Pinning to ``sys.executable`` at install time avoids that silent failure.
    """
    parts = shlex.split(template)
    if parts and parts[0] == "python":
        parts[0] = sys.executable
    return shlex.join(parts)


def install() -> int:
    monocle_hooks = json.loads(_HOOKS_JSON.read_text())["hooks"]
    for hook_groups in monocle_hooks.values():
        for group in hook_groups:
            for hook in group.get("hooks", []):
                if "command" in hook:
                    hook["command"] = _resolve_command(hook["command"])

    _SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings = json.loads(_SETTINGS.read_text()) if _SETTINGS.exists() else {}
    except Exception:
        settings = {}

    if "hooks" not in settings:
        settings["hooks"] = {}

    added, already = [], []
    for event, hook_groups in monocle_hooks.items():
        existing = settings["hooks"].get(event, [])
        # Match on substring so any python interpreter path registering the same
        # module is treated as already registered (avoids duplicates across venvs)
        registered = any(
            _MARKER in h.get("command", "")
            for group in existing
            for h in group.get("hooks", [])
        )
        if registered:
            already.append(event)
        else:
            settings["hooks"].setdefault(event, [])
            settings["hooks"][event].extend(hook_groups)
            added.append(event)

    _SETTINGS.write_text(json.dumps(settings, indent=2))

    if added:
        print(f"[monocle] Registered hooks for: {', '.join(added)}")
    if already:
        print(f"[monocle] Already registered:   {', '.join(already)}")
    print(f"[monocle] Settings written to {_SETTINGS}")
    print(f"[monocle] Hooks bound to interpreter: {sys.executable}")
    print("[monocle] Start a new Claude Code session — traces will flow automatically.")
    return 0
