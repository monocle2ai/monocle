import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Retention for VS Code Copilot Chat sessions. Copilot CLI fires SessionEnd so
# its sessions clean up via cleanup_session() — this TTL is only the safety net
# for VS Code Copilot Chat, which has no SessionEnd hook.
_TTL_SECONDS = 24 * 60 * 60


def _state_dir():
    return Path.cwd() / ".monocle"


def _sessions_dir():
    return _state_dir() / ".copilot_sessions"


def _trace_file():
    return _state_dir() / ".monocle_copilot_trace.jsonl"


def _session_log(session_id):
    sessions = _sessions_dir()
    sessions.mkdir(parents=True, exist_ok=True)
    return sessions / ".monocle_copilot_{}.jsonl".format(session_id)


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


def cleanup_session(session_id):
    """Delete the per-session JSONL log + state file. Called from SessionEnd
    (Copilot CLI fires it; VS Code Copilot Chat doesn't)."""
    from monocle_apptrace.instrumentation.metamodel.github_copilot import git_context

    log = _session_log(session_id)
    state = log.with_suffix(".state.json")
    for f in (log, state):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("cleanup_session: skip %s (%s)", f, e)
    git_context.cleanup(session_id)


def sweep_stale_sessions():
    """Drop session files older than `_TTL_SECONDS`. Called from SessionStart so
    leftover VS Code Copilot Chat sessions (no SessionEnd hook) don't accumulate
    indefinitely. Same pattern codex_cli uses for the same reason."""
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
