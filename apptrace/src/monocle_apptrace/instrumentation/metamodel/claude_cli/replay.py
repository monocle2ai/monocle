"""
Session Replay

Loads per-session event log, processes ONE new turn per call (the turn that
just completed), and emits Monocle spans via the ReplayHandler dummy functions.

Called only on Stop events (one turn = UserPromptSubmit → Stop).

State is tracked in a .state.json sidecar so multi-turn sessions don't
re-emit already-processed turns.
"""

import json
import logging
import os
import sys
from pathlib import Path

from monocle_apptrace.instrumentation.metamodel.claude_cli.trace_events import _session_log

logger = logging.getLogger(__name__)


_telemetry_ready = False


def _configure_telemetry():
    """Run setup_monocle_telemetry once per process.

    Only the hook firings that actually emit spans (Stop, StopFailure, PostCompact,
    SessionEnd-with-pending-replay) need this. Calling it lazily keeps record-only
    hooks (SessionStart, PreToolUse, PostToolUse, …) off the instrumentation hot path.
    """
    global _telemetry_ready
    if _telemetry_ready:
        return
    workflow_name = (
        os.environ.get("MONOCLE_WORKFLOW_NAME")
        or os.environ.get("DEFAULT_WORKFLOW_NAME")
        or "claude-cli"
    )
    from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
    setup_monocle_telemetry(workflow_name=workflow_name)
    _telemetry_ready = True


from monocle_apptrace.instrumentation.metamodel.claude_cli.replay_handlers import ReplayHandler, StopFailureError
from monocle_apptrace.instrumentation.metamodel.claude_cli._helper import (
    build_subagent_tokens,
    read_transcript_tokens,
    read_subagent_transcript,
)
from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION, SPAN_START_TIME, SPAN_END_TIME


# ── State helpers ─────────────────────────────────────────────────────────────

def _state_file(session_id: str) -> Path:
    return _session_log(session_id).with_suffix(".state.json")


def _load_state(session_id: str) -> dict:
    sf = _state_file(session_id)
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except Exception:
            pass
    return {"events_processed": 0, "transcript_lines_processed": 0, "model": "claude"}


def _save_state(session_id: str, state: dict) -> None:
    try:
        _state_file(session_id).write_text(json.dumps(state))
    except Exception:
        pass


# ── Event log ─────────────────────────────────────────────────────────────────

def _load_events(session_id: str) -> list:
    log = _session_log(session_id)
    if not log.exists():
        return []
    events = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


# ── Tool call pairing ─────────────────────────────────────────────────────────

def _pair_tool_call(pre_event: dict, post_event: dict) -> dict:
    is_failure = post_event.get("hook_event_name") == "PostToolUseFailure"
    return {
        "tool_name": pre_event.get("tool_name", post_event.get("tool_name", "")),
        "tool_input": pre_event.get("tool_input", {}),
        "tool_output": post_event.get("tool_response", {}),
        "failed": is_failure,
        "error": post_event.get("error", "") if is_failure else "",
        SPAN_START_TIME: pre_event.get("timestamp"),
        SPAN_END_TIME: post_event.get("timestamp"),
    }


def _pop_pre_tool(store: dict, tool_use_id: str) -> dict:
    """Pop the matching pre-tool event by id, falling back to the first pending."""
    pre = store.pop(tool_use_id, None)
    if pre is None and store:
        pre = store.pop(next(iter(store)))
    return pre


# ── Inference round derivation ────────────────────────────────────────────────

def _derive_inference_rounds(turn_events: list, prompt_ts: str, stop_ts: str, model: str) -> list:
    """Derive inference round timing from hook timestamps.

    Each round spans:
      round 1:   UserPromptSubmit.ts → first PreToolUse.ts
      round N:   last PostToolUse.ts of batch N-1 → first PreToolUse.ts of batch N
      final:     last PostToolUse.ts → Stop.ts   (or whole turn if no tools)
    """
    parent_tools = [
        e for e in turn_events
        if e.get("hook_event_name") in ("PreToolUse", "PostToolUse", "PostToolUseFailure")
        and not e.get("agent_id")
    ]

    if not parent_tools:
        return [{
            SPAN_START_TIME: prompt_ts,
            SPAN_END_TIME: stop_ts,
            "model": model,
            "tokens": {},
            "finish_reason": "end_turn",
            "finish_type": "success",
        }]

    rounds = []
    inference_start = prompt_ts
    last_post_ts = None

    for event in parent_tools:
        name = event["hook_event_name"]
        ts = event.get("timestamp", "")
        if name == "PreToolUse":
            if inference_start is not None:
                rounds.append({
                    SPAN_START_TIME: inference_start,
                    SPAN_END_TIME: ts,
                    "model": model,
                    "tokens": {},
                    "tool_name": event.get("tool_name", ""),
                    "finish_reason": "tool_use",
                    "finish_type": "tool_call",
                })
                inference_start = None
        elif name in ("PostToolUse", "PostToolUseFailure"):
            last_post_ts = ts
            inference_start = ts

    if last_post_ts:
        rounds.append({
            SPAN_START_TIME: last_post_ts,
            SPAN_END_TIME: stop_ts,
            "model": model,
            "tokens": {},
            "finish_reason": "end_turn",
            "finish_type": "success",
        })

    return rounds


# ── Turn processor ────────────────────────────────────────────────────────────

def _collect_turn_events(turn_events: list) -> tuple:
    """Correlate sequential hook events into structured turn data."""
    prompt_event = None
    stop_event = None
    pending_tools: dict = {}
    pending_agent_tools: list = []
    subagent_data: dict = {}
    subagent_order: list = []
    parent_tool_calls: list = []

    for event in turn_events:
        name = event.get("hook_event_name", "")
        agent_id = event.get("agent_id")

        if name == "UserPromptSubmit":
            prompt_event = event

        elif name == "PreToolUse":
            tool_name = event.get("tool_name", "")
            if agent_id:
                sd = subagent_data.setdefault(agent_id, {"pre_tools": {}, "tool_calls": []})
                key = event.get("tool_use_id") or f"_seq_{len(sd['pre_tools'])}"
                sd["pre_tools"][key] = event
            elif tool_name == "Agent":
                pending_agent_tools.append(event)
            else:
                key = event.get("tool_use_id") or f"_seq_{len(pending_tools)}"
                pending_tools[key] = event

        elif name == "SubagentStart":
            sa_id = event.get("agent_id", "")
            sd = subagent_data.setdefault(sa_id, {"pre_tools": {}, "tool_calls": []})
            sd["start_event"] = event
            sd["start_time"] = event.get("timestamp")
            if pending_agent_tools:
                sd["pre_agent_event"] = pending_agent_tools.pop(0)
            if sa_id not in subagent_order:
                subagent_order.append(sa_id)

        elif name == "PostToolUse":
            tool_name = event.get("tool_name", "")
            if agent_id:
                sd = subagent_data.get(agent_id, {})
                pre = _pop_pre_tool(sd.get("pre_tools", {}), event.get("tool_use_id") or "")
                sd.get("tool_calls", []).append(_pair_tool_call(pre or event, event))
            elif tool_name == "Agent":
                resp = event.get("tool_response", {})
                sa_id = resp.get("agentId", "")
                if sa_id and sa_id in subagent_data:
                    subagent_data[sa_id]["post_agent_event"] = event
            else:
                pre = _pop_pre_tool(pending_tools, event.get("tool_use_id") or "")
                parent_tool_calls.append(_pair_tool_call(pre or event, event))

        elif name == "PostToolUseFailure":
            if agent_id:
                sd = subagent_data.get(agent_id, {})
                pre = _pop_pre_tool(sd.get("pre_tools", {}), event.get("tool_use_id") or "")
                sd.get("tool_calls", []).append(_pair_tool_call(pre or event, event))
            else:
                pre = _pop_pre_tool(pending_tools, event.get("tool_use_id") or "")
                parent_tool_calls.append(_pair_tool_call(pre or event, event))

        elif name == "SubagentStop":
            sa_id = event.get("agent_id", "")
            sd = subagent_data.setdefault(sa_id, {"pre_tools": {}, "tool_calls": []})
            sd["stop_event"] = event
            sd["end_time"] = event.get("timestamp")
            sd["last_assistant_message"] = event.get("last_assistant_message", "")

        elif name in ("Stop", "StopFailure"):
            stop_event = event

    return prompt_event, stop_event, parent_tool_calls, subagent_order, subagent_data


def _build_subagents(subagent_order: list, subagent_data: dict, model: str) -> list:
    subagents = []
    for sa_id in subagent_order:
        sd = subagent_data.get(sa_id, {})
        pre_agent = sd.get("pre_agent_event", {})
        post_agent = sd.get("post_agent_event", {})
        post_resp = post_agent.get("tool_response", {}) if post_agent else {}
        subagents.append({
            "agent_id": sa_id,
            "agent_type": sd.get("start_event", {}).get("agent_type", "agent"),
            "prompt": (pre_agent.get("tool_input") or {}).get("prompt", ""),
            "description": (pre_agent.get("tool_input") or {}).get("description", ""),
            "response": sd.get("last_assistant_message", "") or sd.get("stop_event", {}).get("last_assistant_message", ""),
            "tool_calls": sd.get("tool_calls", []),
            "tokens": build_subagent_tokens(post_resp.get("usage", {})),
            "model": model,
            SPAN_START_TIME: sd.get("start_time"),
            SPAN_END_TIME: sd.get("end_time"),
        })
    return subagents


def _process_turn(turn_events: list, session_id: str, model: str, handler: ReplayHandler, transcript_start_line: int = 0) -> None:
    prompt_event, stop_event, parent_tool_calls, subagent_order, subagent_data = _collect_turn_events(turn_events)

    if not prompt_event or not stop_event:
        return

    prompt_ts = prompt_event.get("timestamp", "")
    stop_ts = stop_event.get("timestamp", "")
    inference_rounds = _derive_inference_rounds(turn_events, prompt_ts, stop_ts, model)

    transcript_path = stop_event.get("transcript_path", "")
    parent_tokens = read_transcript_tokens(transcript_path, start_line=transcript_start_line)

    if inference_rounds:
        if parent_tokens:
            inference_rounds[-1]["tokens"] = parent_tokens
        inference_rounds[-1]["output_text"] = stop_event.get("last_assistant_message", "")

    subagents = _build_subagents(subagent_order, subagent_data, model)

    if stop_event.get("hook_event_name") == "StopFailure":
        handler._stop_failure = stop_event.get("error", "")
        handler._stop_failure_details = stop_event.get("error_details", "")
    elif stop_event.get("stop_hook_active"):
        handler._stop_failure = "interrupted"
        handler._stop_failure_details = "Turn was interrupted before completion"

    try:
        handler.handle_turn(
            prompt=prompt_event.get("prompt", ""),
            response=stop_event.get("last_assistant_message", ""),
            tool_calls=parent_tool_calls,
            subagents=subagents,
            inference_rounds=inference_rounds,
            model=model,
            tokens=parent_tokens,
            _turn_start=prompt_ts,
            _turn_end=stop_ts,
            **{
                SPAN_START_TIME: prompt_ts,
                SPAN_END_TIME: stop_ts,
                AGENT_SESSION: session_id,
            },
        )
    except StopFailureError:
        pass  # span already recorded the error via wrapper.py exception flow


# ── Public API ────────────────────────────────────────────────────────────────

def replay_compaction(session_id: str) -> None:
    _configure_telemetry()
    events = _load_events(session_id)

    post = next(
        (e for e in reversed(events) if e.get("hook_event_name") == "PostCompact"),
        None,
    )
    pre = next(
        (e for e in reversed(events) if e.get("hook_event_name") == "PreCompact"),
        None,
    )
    if not post or not pre:
        return

    compact_summary = post.get("compact_summary", "")
    if not compact_summary:
        return

    # The Haiku compaction agent fires SubagentStop between PreCompact and PostCompact.
    # Its agent_transcript_path carries the real model name and exact token counts.
    pre_idx = next((i for i, e in enumerate(events) if e is pre), 0)
    post_idx = next((i for i, e in enumerate(events) if e is post), len(events))
    compaction_stop = next(
        (e for e in events[pre_idx:post_idx] if e.get("hook_event_name") == "SubagentStop"),
        None,
    )

    model = _load_state(session_id).get("model", "claude")
    tokens = {}
    if compaction_stop:
        agent_path = compaction_stop.get("agent_transcript_path", "")
        model, tokens = read_subagent_transcript(agent_path)

    if not tokens:
        tokens = read_transcript_tokens(pre.get("transcript_path", ""))

    logger.debug(f"--- Compaction replay for session {session_id} ---")
    handler = ReplayHandler()
    handler.handle_inference_round(
        input_text="",
        output_text=compact_summary,
        model=model,
        tokens=tokens,
        finish_reason="compaction",
        finish_type=post.get("trigger", "manual"),
        **{
            SPAN_START_TIME: pre.get("timestamp"),
            SPAN_END_TIME: post.get("timestamp"),
            AGENT_SESSION: session_id,
        },
    )
    logger.debug("--- Compaction replay done ---")


def cleanup_session(session_id: str) -> None:
    """Delete per-session working files after a SessionEnd.

    Replays any unprocessed events first so no turn is silently dropped
    (guards against crashes where Stop was missed before SessionEnd fired).
    Also prunes the session's subagent IDs from the shared registry.
    """
    from monocle_apptrace.instrumentation.metamodel.claude_cli.trace_events import unmark_subagent_sessions

    state = _load_state(session_id)
    events = _load_events(session_id)

    if state["events_processed"] < len(events):
        logger.debug(f"SessionEnd: {len(events) - state['events_processed']} unprocessed events found — replaying before cleanup")
        replay_session(session_id)

    subagent_ids = [
        e.get("agent_id") for e in events
        if e.get("hook_event_name") == "SubagentStart" and e.get("agent_id")
    ]
    unmark_subagent_sessions(subagent_ids)

    _session_log(session_id).unlink(missing_ok=True)
    _state_file(session_id).unlink(missing_ok=True)
    logger.debug(f"SessionEnd: cleaned up session files for {session_id}")


def replay_session(session_id: str) -> None:
    state = _load_state(session_id)
    events = _load_events(session_id)
    new_events = events[state["events_processed"]:]

    if not new_events:
        return

    _configure_telemetry()

    model = state.get("model", "claude")
    for e in events:
        if e.get("hook_event_name") == "SessionStart" and e.get("model"):
            model = e["model"]
            break

    stop_event = next((e for e in reversed(new_events) if e.get("hook_event_name") in ("Stop", "StopFailure")), None)
    transcript_path = stop_event.get("transcript_path", "") if stop_event else ""
    transcript_start_line = state.get("transcript_lines_processed", 0)
    transcript_line_count = 0
    if transcript_path:
        try:
            transcript_line_count = len(Path(transcript_path).read_text(encoding="utf-8", errors="replace").splitlines())
        except Exception:
            pass

    logger.debug(f"--- Replay: {len(new_events)} new events for session {session_id} ---")
    handler = ReplayHandler()
    _process_turn(new_events, session_id, model, handler, transcript_start_line=transcript_start_line)

    state["events_processed"] = len(events)
    state["transcript_lines_processed"] = transcript_line_count or transcript_start_line
    state["model"] = model
    _save_state(session_id, state)
    logger.debug("--- Replay done ---")


def main() -> None:
    if len(sys.argv) != 2:
        logger.debug("Usage: replay.py <session_id>")
        sys.exit(1)
    replay_session(sys.argv[1])


if __name__ == "__main__":
    main()
