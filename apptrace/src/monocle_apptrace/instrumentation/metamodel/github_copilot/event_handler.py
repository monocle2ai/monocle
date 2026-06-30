#!/usr/bin/env python3
"""Hook entry point for GitHub Copilot (VS Code Copilot Chat + Copilot CLI)."""

import json
import logging
import sys
from datetime import datetime, timezone

from monocle_apptrace.instrumentation.metamodel.github_copilot.trace_events import (
    _session_log,
    record_trace_event,
    mark_subagent_session,
    is_subagent_session,
    cleanup_session,
    sweep_stale_sessions,
)
from monocle_apptrace.instrumentation.metamodel.github_copilot._otel_tokens import prune_otel_file
from monocle_apptrace.instrumentation.metamodel.github_copilot import git_context

logger = logging.getLogger(__name__)

_REPLAY_TRIGGERS = {"Stop"}
_COMPACTION_TRIGGERS = {"PreCompact"}


def _event_cwd(event_data: dict) -> str:
    for key in (
        "cwd",
        "working_directory",
        "workingDirectory",
        "workspace_root",
        "workspaceRoot",
        "project_dir",
        "projectDir",
        "project_root",
        "projectRoot",
    ):
        value = event_data.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _normalize(event: dict) -> dict:
    """Canonicalize both VS Code Copilot Chat (camelCase, flat tool_response)
    and Copilot CLI (nested tool_result, agent_name/agent_display_name) into
    one snake_case shape so the rest of the metamodel is schema-agnostic."""
    if "session_id" not in event and "sessionId" in event:
        event["session_id"] = event["sessionId"]
    if "hook_event_name" not in event and "hookEventName" in event:
        event["hook_event_name"] = event["hookEventName"]
    if "tool_use_id" not in event and "toolUseId" in event:
        event["tool_use_id"] = event["toolUseId"]
    if "tool_response" not in event and isinstance(event.get("tool_result"), dict):
        event["tool_response"] = event["tool_result"].get("text_result_for_llm", "")
    if "agent_id" not in event and "agent_name" in event:
        event["agent_id"] = event["agent_name"]
    if "agent_type" not in event and "agent_display_name" in event:
        event["agent_type"] = event["agent_display_name"]
    return event


def record_event(event_data: dict) -> None:
    session_id = event_data.get("session_id", "unknown")
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **event_data}
    with _session_log(session_id).open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    record_trace_event(entry)


def _shutdown_emitted_for(session_id: str) -> bool:
    state_f = _session_log(session_id).with_suffix(".state.json")
    try:
        return bool(json.loads(state_f.read_text()).get("shutdown_emitted"))
    except Exception:
        return False


def _retry_pending_copilot_cli_sessions() -> None:
    """Copilot CLI writes `session.shutdown` to the transcript a few ms AFTER
    the SessionEnd hook returns, so the synchronous replay-on-SessionEnd misses
    it. Catch up here on the next SessionStart, when those transcripts are
    stable. No-op for VS Code Copilot Chat (replay_session early-exits)."""
    from monocle_apptrace.instrumentation.metamodel.github_copilot.trace_events import _sessions_dir
    from monocle_apptrace.instrumentation.metamodel.github_copilot.replay import replay_session
    sessions = _sessions_dir()
    if not sessions.exists():
        return
    prefix = ".monocle_copilot_"
    for f in sessions.iterdir():
        if not (f.is_file() and f.name.startswith(prefix) and f.name.endswith(".jsonl")):
            continue
        sid = f.name[len(prefix):-len(".jsonl")]
        state_f = f.with_suffix(".state.json")
        if not state_f.exists():
            continue
        try:
            if json.loads(state_f.read_text()).get("shutdown_emitted"):
                continue
        except Exception:
            continue
        try:
            replay_session(sid)
        except Exception as e:
            logger.debug(f"retry_pending: replay({sid}) failed: {e}")
            continue
        if _shutdown_emitted_for(sid):
            cleanup_session(sid)


def main() -> None:
    raw = sys.stdin.read()
    try:
        event_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug(f"ERROR: could not parse stdin as JSON – {exc}")
        sys.exit(1)

    event_data = _normalize(event_data)
    session_id = event_data.get("session_id", "unknown")
    event_name = event_data.get("hook_event_name", "unknown")

    if event_name == "SessionStart":
        sweep_stale_sessions()
        prune_otel_file()
        _retry_pending_copilot_cli_sessions()

    record_event(event_data)
    logger.debug(f"Recorded {event_name} for session {session_id}")

    if event_name == "SubagentStart":
        agent_id = event_data.get("agent_id", "")
        if agent_id:
            mark_subagent_session(agent_id)

    if event_name == "UserPromptSubmit" and not is_subagent_session(session_id):
        git_context.capture_turn_baseline(session_id, cwd=_event_cwd(event_data) or None)

    if event_name in _REPLAY_TRIGGERS:
        if is_subagent_session(session_id):
            logger.debug(f"Skipping replay for subagent session {session_id}")
        else:
            from monocle_apptrace.instrumentation.metamodel.github_copilot.replay import replay_session
            try:
                replay_session(session_id)
            except Exception as e:
                logger.debug(f"Replay error: {e}")

    if event_name in _COMPACTION_TRIGGERS:
        from monocle_apptrace.instrumentation.metamodel.github_copilot.replay import replay_compaction
        try:
            replay_compaction(session_id)
        except Exception as e:
            logger.debug(f"Compaction replay error: {e}")

    if event_name == "SessionEnd":
        try:
            from monocle_apptrace.instrumentation.metamodel.github_copilot.replay import replay_session
            replay_session(session_id)
        except Exception as e:
            logger.debug(f"Replay-on-SessionEnd error: {e}")
        # Skip cleanup if shutdown wasn't emitted; next SessionStart retries.
        if _shutdown_emitted_for(session_id):
            cleanup_session(session_id)

    sys.stdout.write("{}")


if __name__ == "__main__":
    main()
