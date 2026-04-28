#!/usr/bin/env python3
"""
Claude CLI Event Handler

Hook entry point for all Claude Code CLI events.
  - Reads event JSON from stdin
  - Appends every event with a UTC timestamp to the per-session JSONL log
  - On Stop: triggers replay to emit Monocle spans for the completed turn
"""

import json
import logging
import sys
from datetime import datetime, timezone

from monocle_apptrace.instrumentation.metamodel.claude_cli.trace_events import (
    _session_log,
    record_trace_event,
    mark_subagent_session,
    is_subagent_session,
)

logger = logging.getLogger(__name__)
from monocle_apptrace.instrumentation.metamodel.claude_cli.replay import replay_session, replay_compaction, cleanup_session

# Events that trigger span emission for the completed turn
_REPLAY_TRIGGERS = {"Stop", "StopFailure"}
_COMPACTION_TRIGGERS = {"PostCompact"}


def record_event(event_data: dict) -> None:
    session_id = event_data.get("session_id", "unknown")
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **event_data}
    with _session_log(session_id).open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    record_trace_event(entry)


def main() -> None:
    raw = sys.stdin.read()
    try:
        event_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug(f"ERROR: could not parse stdin as JSON – {exc}")
        sys.exit(1)

    session_id = event_data.get("session_id", "unknown")
    event_name = event_data.get("hook_event_name", "unknown")

    record_event(event_data)
    logger.debug(f"Recorded {event_name} for session {session_id}")

    # Track subagent sessions so we don't emit duplicate top-level turns for them
    if event_name == "SubagentStart":
        agent_id = event_data.get("agent_id", "")
        if agent_id:
            mark_subagent_session(agent_id)

    if event_name in _REPLAY_TRIGGERS:
        if is_subagent_session(session_id):
            logger.debug(f"Skipping replay for subagent session {session_id} (captured in parent turn)")
        else:
            replay_session(session_id)

    if event_name in _COMPACTION_TRIGGERS:
        replay_compaction(session_id)

    if event_name == "SessionEnd":
        cleanup_session(session_id)


if __name__ == "__main__":
    main()
