"""
Per-session state for replay (cursor into the Codex transcript).

We don't capture our own per-event JSONL anymore — Codex's transcript at
``transcript_path`` is the source of truth. Only the cursor sidecar lives here.
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_DIR = Path(os.getcwd()) / ".monocle"
SESSIONS_DIR = _STATE_DIR / ".codex_sessions"

_TTL_SECONDS = 24 * 60 * 60


def _state_file(session_id: str) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f".monocle_codex_{session_id}.state.json"


def load_state(session_id: str) -> dict:
    sf = _state_file(session_id)
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except Exception:
            pass
    return {"transcript_lines_processed": 0, "model": "codex"}


def save_state(session_id: str, state: dict) -> None:
    try:
        _state_file(session_id).write_text(json.dumps(state))
    except Exception:
        pass


def sweep_stale_sessions() -> None:
    """Drop session state files older than TTL.

    Codex has no SessionEnd hook, so SessionStart is the natural cleanup point.
    """
    if not SESSIONS_DIR.exists():
        return
    cutoff = time.time() - _TTL_SECONDS
    for f in SESSIONS_DIR.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
        except Exception as e:
            logger.debug(f"Sweep skipped {f}: {e}")
