import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _state_dir():
    return Path.cwd() / ".monocle"


def _sessions_dir():
    return _state_dir() / ".claude_sessions"


def _trace_file():
    return _state_dir() / ".monocle_claude_trace.jsonl"


def _session_log(session_id):
    sessions = _sessions_dir()
    sessions.mkdir(parents=True, exist_ok=True)
    return sessions / ".monocle_claude_{}.jsonl".format(session_id)


def record_trace_event(entry):
    """Append an enriched event dict as a single JSON line to the trace file."""
    sessions = _sessions_dir()
    sessions.mkdir(parents=True, exist_ok=True)
    with _trace_file().open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _subagent_sessions_file():
    sessions = _sessions_dir()
    sessions.mkdir(parents=True, exist_ok=True)
    return sessions / ".subagent_sessions.json"


def mark_subagent_session(agent_id):
    """Record that agent_id is a subagent session so we skip emitting a top-level turn for it."""
    f = _subagent_sessions_file()
    try:
        known = json.loads(f.read_text()) if f.exists() else []
    except Exception as e:
        logger.debug("Error reading subagent sessions file: %s", e)
        known = []
    if agent_id not in known:
        known.append(agent_id)
        f.write_text(json.dumps(known))


def is_subagent_session(session_id):
    """Return True if session_id was spawned as a subagent by some parent session."""
    f = _subagent_sessions_file()
    try:
        return session_id in (json.loads(f.read_text()) if f.exists() else [])
    except Exception as e:
        logger.debug("Error reading subagent sessions file: %s", e)
        return False


def unmark_subagent_sessions(agent_ids):
    """Remove agent_ids from the subagent sessions registry once the parent session ends."""
    if not agent_ids:
        return
    f = _subagent_sessions_file()
    try:
        known = json.loads(f.read_text()) if f.exists() else []
        pruned = [aid for aid in known if aid not in agent_ids]
        f.write_text(json.dumps(pruned))
    except Exception as e:
        logger.debug("Error updating subagent sessions file: %s", e)
