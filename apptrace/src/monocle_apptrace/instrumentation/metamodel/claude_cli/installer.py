import json
from pathlib import Path

_COMMAND = "python -m monocle_apptrace.instrumentation.metamodel.claude_cli"
_MARKER = "monocle_apptrace.instrumentation.metamodel.claude_cli"
_SETTINGS = Path.home() / ".claude" / "settings.json"
_HOOKS_JSON = Path(__file__).parent / "hooks.json"


def install() -> int:
    monocle_hooks = json.loads(_HOOKS_JSON.read_text())["hooks"]

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
    print("[monocle] Start a new Claude Code session — traces will flow automatically.")
    return 0
