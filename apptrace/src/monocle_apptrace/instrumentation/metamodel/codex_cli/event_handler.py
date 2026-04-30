#!/usr/bin/env python3
"""Hook entry point. SessionStart sweeps stale state; Stop triggers replay."""

import json
import logging
import sys

from monocle_apptrace.instrumentation.metamodel.codex_cli.replay import replay_session
from monocle_apptrace.instrumentation.metamodel.codex_cli.trace_events import sweep_stale_sessions

logger = logging.getLogger(__name__)


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
        try:
            replay_session(data.get("session_id"), data.get("transcript_path"))
        except Exception as e:
            logger.debug(f"Replay error: {e}")
        # Stop expects JSON on stdout per Codex docs
        sys.stdout.write("{}")


if __name__ == "__main__":
    main()
