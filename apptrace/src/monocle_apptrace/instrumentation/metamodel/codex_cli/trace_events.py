import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_TTL_SECONDS = 24 * 60 * 60


def _state_dir():
    return Path.cwd() / ".monocle"


def _sessions_dir():
    return _state_dir() / ".codex_sessions"


def _state_file(session_id):
    sessions = _sessions_dir()
    sessions.mkdir(parents=True, exist_ok=True)
    return sessions / ".monocle_codex_{}.state.json".format(session_id)


def load_state(session_id):
    sf = _state_file(session_id)
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except Exception:
            pass
    return {"transcript_lines_processed": 0, "model": "codex"}


def save_state(session_id, state):
    try:
        _state_file(session_id).write_text(json.dumps(state))
    except Exception:
        pass


def sweep_stale_sessions():
    """Drop session state files older than TTL (no SessionEnd hook in Codex)."""
    sessions = _sessions_dir()
    if not sessions.exists():
        return
    cutoff = time.time() - _TTL_SECONDS
    for f in sessions.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
        except Exception as e:
            logger.debug("Sweep skipped %s: %s", f, e)
