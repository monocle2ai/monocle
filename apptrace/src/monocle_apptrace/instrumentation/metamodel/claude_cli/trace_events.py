"""
Monocle Claude Trace

Per-session JSONL event logs and the append-only trace file live in a
``.monocle/`` folder at the directory where Claude Code was launched, so each
project keeps its own trace history alongside its source.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_DIR = Path(os.getcwd()) / ".monocle"
SESSIONS_DIR = _STATE_DIR / ".claude_sessions"
TRACE_FILE = _STATE_DIR / ".monocle_claude_trace.jsonl"


def _session_log(session_id: str) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f".monocle_claude_{session_id}.jsonl"


def record_trace_event(entry: dict) -> None:
    """Append an enriched event dict as a single JSON line to the trace file."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    with TRACE_FILE.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _subagent_sessions_file() -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / ".subagent_sessions.json"


def mark_subagent_session(agent_id: str) -> None:
    """Record that agent_id is a subagent session so we skip emitting a top-level turn for it."""
    f = _subagent_sessions_file()
    try:
        known = json.loads(f.read_text()) if f.exists() else []
    except Exception as e:
        logger.debug(f"Error reading subagent sessions file: {e}")
        known = []
    if agent_id not in known:
        known.append(agent_id)
        f.write_text(json.dumps(known))


def is_subagent_session(session_id: str) -> bool:
    """Return True if session_id was spawned as a subagent by some parent session."""
    f = _subagent_sessions_file()
    try:
        return session_id in (json.loads(f.read_text()) if f.exists() else [])
    except Exception as e:
        logger.debug(f"Error reading subagent sessions file: {e}")
        return False


def unmark_subagent_sessions(agent_ids: list) -> None:
    """Remove agent_ids from the subagent sessions registry once the parent session ends."""
    if not agent_ids:
        return
    f = _subagent_sessions_file()
    try:
        known = json.loads(f.read_text()) if f.exists() else []
        pruned = [aid for aid in known if aid not in agent_ids]
        f.write_text(json.dumps(pruned))
    except Exception as e:
        logger.debug(f"Error updating subagent sessions file: {e}")
