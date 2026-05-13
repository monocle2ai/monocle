"""Hook entry point. SessionStart sweeps stale state; Stop triggers replay."""

import json
import logging
import sys

from monocle_apptrace.instrumentation.metamodel.copilot_cli.replay import replay_session
from monocle_apptrace.instrumentation.metamodel.copilot_cli.trace_events import sweep_stale_sessions

logger = logging.getLogger(__name__)


def main() -> None:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug("ERROR: could not parse stdin as JSON – %s", exc)
        sys.exit(1)

    # Copilot CLI uses camelCase; normalise to match existing hook patterns
    event_name = data.get("hook_event_name") or data.get("hookEventName", "")
    session_id = data.get("session_id") or data.get("sessionId", "")

    if event_name == "SessionStart":
        sweep_stale_sessions()
        return

    if event_name == "Stop":
        if not session_id:
            logger.debug("Stop event missing session_id")
            sys.stdout.write("{}")
            return
        try:
            replay_session(session_id)
        except Exception as e:
            logger.debug("Replay error: %s", e)
        # Copilot CLI Stop hook expects JSON on stdout
        sys.stdout.write("{}")


if __name__ == "__main__":
    main()
