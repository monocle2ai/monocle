#!/usr/bin/env python3
"""Hook entry point. SessionStart sweeps stale state; Stop triggers replay."""

import json
import logging
import sys

from monocle_apptrace.instrumentation.metamodel.codex_cli.replay import replay_session
from monocle_apptrace.instrumentation.metamodel.codex_cli.trace_events import sweep_stale_sessions

logger = logging.getLogger(__name__)


def _is_subagent_session(transcript_path) -> bool:
    """Codex fires a Stop hook for every thread, including spawned subagents.
    The parent's replay already captures them as Sub-Agent spans, so skip these
    to avoid duplicate trace trees with distinct scope.agentic.session values."""
    if not transcript_path:
        return False
    try:
        with open(transcript_path) as fh:
            meta = json.loads(fh.readline()).get("payload", {})
    except Exception:
        return False
    source = meta.get("source")
    return isinstance(source, dict) and "subagent" in source


def main() -> None:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug(f"ERROR: could not parse stdin as JSON – {exc}")
        sys.exit(1)

    name = data.get("hook_event_name", "")

    if name == "SessionStart":
        sweep_stale_sessions()
        return

    if name == "Stop":
        transcript = data.get("transcript_path")
        if _is_subagent_session(transcript):
            logger.debug(f"Skipping subagent Stop for {data.get('session_id')}")
        else:
            try:
                replay_session(data.get("session_id"), transcript)
            except Exception as e:
                logger.debug(f"Replay error: {e}")
        # Stop expects JSON on stdout per Codex docs
        sys.stdout.write("{}")


if __name__ == "__main__":
    main()
