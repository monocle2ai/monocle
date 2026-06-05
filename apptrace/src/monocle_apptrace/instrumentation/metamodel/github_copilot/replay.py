"""Replay a GitHub Copilot session's transcript into Monocle spans on Stop."""

import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

try:
    import fcntl  # POSIX only
except ImportError:
    fcntl = None

from monocle_apptrace.instrumentation.common.constants import (
    AGENT_SESSION,
    SPAN_START_TIME,
    SPAN_END_TIME,
)
from monocle_apptrace.instrumentation.metamodel.github_copilot.trace_events import (
    _session_log,
    _sessions_dir,
)
from monocle_apptrace.instrumentation.metamodel.github_copilot.replay_handlers import ReplayHandler
from monocle_apptrace.instrumentation.metamodel.github_copilot._otel_tokens import lookup_turn_tokens

logger = logging.getLogger(__name__)

_SUBAGENT_LAUNCHER_TOOL_NAMES = {"runSubagent"}  # filtered from parent tool list

_telemetry_ready = False


def _configure_telemetry():
    global _telemetry_ready
    if _telemetry_ready:
        return
    from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value
    from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
    workflow_name = get_monocle_env_value("MONOCLE_WORKFLOW_NAME") or Path.cwd().name
    setup_monocle_telemetry(workflow_name=workflow_name)
    _telemetry_ready = True


@contextmanager
def _session_lock(session_id: str):
    """Advisory file lock so concurrent Stop hooks don't double-emit."""
    sessions = _sessions_dir()
    sessions.mkdir(parents=True, exist_ok=True)
    lock_f = (sessions / f".{session_id}.lock").open("w")
    try:
        if fcntl is not None:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_f.close()


def _load_jsonl(path) -> list:
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        return []
    events = []
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.debug(f"Error reading {p}: {e}")
    return events


def _state_file(session_id: str) -> Path:
    return _session_log(session_id).with_suffix(".state.json")


def _load_state(session_id: str) -> dict:
    sf = _state_file(session_id)
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except Exception:
            pass
    return {"interactions_processed": [], "model": "copilot"}


def _save_state(session_id: str, state: dict) -> None:
    try:
        _state_file(session_id).write_text(json.dumps(state))
    except Exception:
        pass


def _latest_transcript_path(hook_events: list) -> Optional[str]:
    for e in reversed(hook_events):
        tp = e.get("transcript_path")
        if tp:
            return tp
    return None


def _build_subagent_intervals(hook_events: list) -> list:
    """Pair SubagentStart/SubagentStop into [start_ts, end_ts] windows. Events in
    those windows belong to the subagent, not the parent turn."""
    intervals = []
    pending = {}
    for e in hook_events:
        name = e.get("hook_event_name")
        aid = e.get("agent_id", "")
        if name == "SubagentStart" and aid:
            pending[aid] = e
        elif name == "SubagentStop" and aid and aid in pending:
            start = pending.pop(aid)
            intervals.append({
                "agent_id": aid,
                "agent_type": start.get("agent_type", "agent") or e.get("agent_type", "agent"),
                "start_ts": start.get("timestamp", ""),
                "end_ts": e.get("timestamp", ""),
            })
    if pending and hook_events:
        last_ts = hook_events[-1].get("timestamp", "")
        for aid, start in pending.items():
            intervals.append({
                "agent_id": aid,
                "agent_type": start.get("agent_type", "agent"),
                "start_ts": start.get("timestamp", ""),
                "end_ts": last_ts,
            })
    return intervals


def _interval_for_ts(ts: str, intervals: list) -> Optional[dict]:
    if not ts:
        return None
    for itv in intervals:
        if itv["start_ts"] <= ts <= itv["end_ts"]:
            return itv
    return None


def _index_tool_outputs(hook_events: list, intervals: list) -> tuple:
    """PreToolUse/PostToolUse `agent_id` is empty in practice, so we route by
    timestamp into either parent or matching subagent. PostToolUseFailure (Copilot
    CLI only) carries the error message the transcript doesn't expose."""
    parent_outputs = []
    subagent_outputs = {itv["agent_id"]: [] for itv in intervals}
    for e in hook_events:
        name = e.get("hook_event_name")
        if name not in ("PostToolUse", "PostToolUseFailure"):
            continue
        ts = e.get("timestamp", "")
        is_failure = name == "PostToolUseFailure"
        record = {
            "tool_name": e.get("tool_name", ""),
            "tool_response": e.get("tool_response", "") if not is_failure else "",
            "failed": is_failure,
            "error": e.get("error", "") if is_failure else "",
            "timestamp": ts,
        }
        itv = _interval_for_ts(ts, intervals)
        (subagent_outputs[itv["agent_id"]] if itv else parent_outputs).append(record)
    return parent_outputs, subagent_outputs


def _walk_interactions(transcript_events: list, subagent_intervals: list) -> list:
    """Walk the transcript into interactions, nesting subagent activity inside
    the parent turn. A `user.message` whose timestamp falls inside a subagent
    interval is the subagent's task prompt, not a new top-level turn."""
    interactions = []
    current = None
    current_subagent = None
    pending_tools = {}          # toolCallId -> (container, start_dict)
    current_model = "copilot"   # updated by session.start.model / session.model_change

    def _container_for_ts(ts: str):
        nonlocal current_subagent
        itv = _interval_for_ts(ts, subagent_intervals)
        if itv is None:
            current_subagent = None
            return current
        if current is None:
            return None
        if current_subagent is None or current_subagent.get("agent_id") != itv["agent_id"]:
            current_subagent = {
                "agent_id": itv["agent_id"],
                "agent_type": itv["agent_type"],
                "prompt": "",
                "assistant_messages": [],
                "tool_calls": [],
                "model": current_model,
                "start_ts": itv["start_ts"],
                "end_ts": itv["end_ts"],
            }
            current.setdefault("subagents", []).append(current_subagent)
        return current_subagent

    for event in transcript_events:
        etype = event.get("type", "")
        data = event.get("data") or {}
        ts = event.get("timestamp", "")

        if etype == "session.start":
            if data.get("model"):
                current_model = data["model"]
            continue
        if etype == "session.model_change":
            new_model = data.get("newModel") or data.get("model")
            if new_model:
                current_model = new_model
            continue

        if etype == "user.message":
            itv = _interval_for_ts(ts, subagent_intervals)
            if itv and current is not None:
                container = _container_for_ts(ts)
                if container is not None:
                    container["prompt"] = data.get("content", "")
                continue
            if current:
                interactions.append(current)
            current = {
                "id": event.get("id", ""),
                "prompt": data.get("content", ""),
                "assistant_messages": [],
                "tool_calls": [],
                "subagents": [],
                "model": current_model,
                "turn_start": ts,
                "turn_end": ts,
            }
            current_subagent = None

        elif etype == "assistant.message" and current is not None:
            container = _container_for_ts(ts)
            if container is None:
                continue
            container.setdefault("assistant_messages", []).append({
                "content": data.get("content", "") or "",
                "reasoning": data.get("reasoningText", "") or "",
                "tool_requests": data.get("toolRequests") or [],
                "timestamp": ts,
            })
            current["turn_end"] = ts

        elif etype == "tool.execution_start" and current is not None:
            call_id = data.get("toolCallId", "")
            if not call_id:
                continue
            container = _container_for_ts(ts)
            if container is None:
                continue
            pending_tools[call_id] = (container, {
                "tool_name": data.get("toolName", ""),
                "arguments": data.get("arguments", ""),
                "start_ts": ts,
            })

        elif etype == "tool.execution_complete" and current is not None:
            call_id = data.get("toolCallId", "")
            pre = pending_tools.pop(call_id, None)
            if pre:
                container, pre_data = pre
                # Drop launcher tool — Sub-Agent span captures the same content.
                if pre_data["tool_name"] in _SUBAGENT_LAUNCHER_TOOL_NAMES and container is current:
                    current["turn_end"] = ts
                    continue
                container.setdefault("tool_calls", []).append({
                    "call_id": call_id,
                    "tool_name": pre_data["tool_name"],
                    "arguments": pre_data["arguments"],
                    "success": bool(data.get("success", True)),
                    SPAN_START_TIME: pre_data["start_ts"],
                    SPAN_END_TIME: ts,
                })
                current["turn_end"] = ts

        elif etype == "assistant.turn_end" and current is not None:
            current["turn_end"] = ts

    if current:
        interactions.append(current)
    for itr in interactions:
        if not itr["assistant_messages"]:
            itr["interrupted"] = True
    return interactions


def _parse_args(raw_args) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"_raw": raw_args}
    return {}


def _build_inference_round(interaction: dict, prompt: str) -> dict:
    """One inference round per turn — model continuations collapse together."""
    msgs = interaction.get("assistant_messages", [])
    if not msgs:
        return {}
    first_msg, last_msg = msgs[0], msgs[-1]
    final_content = next((m["content"] for m in reversed(msgs) if m.get("content")), "")
    dispatched_tools = [
        r.get("name") for m in msgs for r in (m.get("tool_requests") or []) if r.get("name")
    ]
    finish_reason = "tool_use" if dispatched_tools else "end_turn"
    finish_type = "tool_call" if dispatched_tools else "success"
    return {
        SPAN_START_TIME: interaction.get("turn_start") or first_msg["timestamp"],
        SPAN_END_TIME: last_msg["timestamp"],
        "model": interaction.get("model", "copilot"),
        "tokens": {},
        "input_text": prompt,
        "output_text": final_content,
        "finish_reason": finish_reason,
        "finish_type": finish_type,
    }


def _hydrate_tool_outputs(tool_calls: list, scope_outputs: list) -> None:
    """Hook tool_use_ids don't match transcript toolCallIds, so match by name in
    sequence. PostToolUseFailure propagates `failed`/`error` over the transcript's
    success flag."""
    available = list(scope_outputs)
    for tc in tool_calls:
        match_idx = next(
            (i for i, out in enumerate(available) if out["tool_name"] == tc["tool_name"]),
            None,
        )
        if match_idx is not None:
            out = available.pop(match_idx)
            tc["tool_output"] = out["tool_response"]
            if out.get("failed"):
                tc["failed"] = True
                tc["error"] = out.get("error", "")
        else:
            tc["tool_output"] = ""


def _convert_tool_calls(transcript_tool_calls: list, outputs: list) -> list:
    _hydrate_tool_outputs(transcript_tool_calls, outputs)
    out = []
    for tc in transcript_tool_calls:
        failed = tc.get("failed", False) or not tc.get("success", True)
        out.append({
            "tool_name": tc["tool_name"],
            "tool_input": _parse_args(tc.get("arguments", "")),
            "tool_output": tc.get("tool_output", ""),
            "call_id": tc.get("call_id", ""),
            "failed": failed,
            "error": tc.get("error", ""),
            SPAN_START_TIME: tc.get(SPAN_START_TIME),
            SPAN_END_TIME: tc.get(SPAN_END_TIME),
        })
    return out


def _build_subagent_records(interaction: dict, subagent_outputs_by_id: dict) -> list:
    records = []
    for sa in interaction.get("subagents") or []:
        outputs = subagent_outputs_by_id.get(sa["agent_id"], [])
        sa_tool_calls = _convert_tool_calls(sa.get("tool_calls", []), outputs)
        final_response = next(
            (m["content"] for m in reversed(sa.get("assistant_messages", [])) if m.get("content")),
            "",
        )
        records.append({
            "agent_id": sa["agent_id"],
            "agent_type": sa.get("agent_type", "agent"),
            "prompt": sa.get("prompt", ""),
            "description": sa.get("prompt", "")[:120],
            "response": final_response,
            "tool_calls": sa_tool_calls,
            "tokens": {},
            "model": sa.get("model", "copilot"),
            SPAN_START_TIME: sa.get("start_ts"),
            SPAN_END_TIME: sa.get("end_ts"),
        })
    return records


def replay_session(session_id: str) -> None:
    if not session_id:
        return
    with _session_lock(session_id):
        _replay_session_locked(session_id)


def _replay_session_locked(session_id: str) -> None:
    hook_events = _load_jsonl(_session_log(session_id))
    if not hook_events:
        return
    transcript_path = _latest_transcript_path(hook_events)
    if not transcript_path:
        logger.debug(f"No transcript_path for {session_id}")
        return
    transcript_events = _load_jsonl(transcript_path)
    if not transcript_events:
        return

    subagent_intervals = _build_subagent_intervals(hook_events)
    interactions = _walk_interactions(transcript_events, subagent_intervals)

    state = _load_state(session_id)
    already = set(state.get("interactions_processed", []))
    # Skip interactions with no assistant response and not flagged interrupted (still mid-flight).
    new_interactions = [
        i for i in interactions
        if i["id"] not in already and (i.get("assistant_messages") or i.get("interrupted"))
    ]

    if not new_interactions:
        return

    parent_outputs, subagent_outputs = _index_tool_outputs(hook_events, subagent_intervals)
    _configure_telemetry()
    handler = ReplayHandler()

    for interaction in new_interactions:
        iid = interaction["id"]
        prompt = interaction["prompt"]
        turn_start = interaction.get("turn_start", "")
        turn_end = interaction.get("turn_end", turn_start)
        tool_calls = _convert_tool_calls(interaction.get("tool_calls", []), parent_outputs)
        subagents = _build_subagent_records(interaction, subagent_outputs)
        response = next(
            (m["content"] for m in reversed(interaction.get("assistant_messages", [])) if m.get("content")),
            "",
        )
        round_dict = _build_inference_round(interaction, prompt)
        turn_tokens = {}
        turn_model = interaction.get("model", "copilot")
        # Token counts come from Copilot's own OTel export — both VS Code Chat and
        # the CLI write it — anchored by trace id within the turn window.
        if round_dict:
            otel_tokens, otel_trace_id, otel_model = lookup_turn_tokens(
                round_dict[SPAN_START_TIME], round_dict[SPAN_END_TIME]
            )
            if otel_tokens:
                round_dict["tokens"] = otel_tokens
                round_dict["otel_trace_id"] = otel_trace_id
                if otel_model:
                    round_dict["model"] = otel_model
                turn_tokens = otel_tokens
                turn_model = round_dict["model"]
        try:
            handler.handle_turn(
                prompt=prompt,
                response=response,
                tool_calls=tool_calls,
                subagents=subagents,
                inference_rounds=[round_dict] if round_dict else [],
                model=turn_model,
                tokens=turn_tokens,
                _turn_start=turn_start,
                _turn_end=turn_end,
                **{SPAN_START_TIME: turn_start, SPAN_END_TIME: turn_end, AGENT_SESSION: session_id},
            )
        except Exception as e:
            logger.debug(f"Turn replay failed for {iid}: {e}")
        already.add(iid)

    state["interactions_processed"] = list(already)
    _save_state(session_id, state)


def replay_compaction(session_id: str) -> None:
    """VS Code fires only PreCompact (no PostCompact). Emit zero-duration span."""
    hook_events = _load_jsonl(_session_log(session_id))
    pre = next(
        (e for e in reversed(hook_events) if e.get("hook_event_name") == "PreCompact"),
        None,
    )
    if not pre:
        return
    _configure_telemetry()
    ts = pre.get("timestamp", "")
    ReplayHandler().handle_inference_round(
        input_text="",
        output_text="",
        model=_load_state(session_id).get("model", "copilot"),
        tokens={},
        finish_reason="compaction",
        finish_type=pre.get("trigger", "auto"),
        **{SPAN_START_TIME: ts, SPAN_END_TIME: ts, AGENT_SESSION: session_id},
    )


def main() -> None:
    if len(sys.argv) != 2:
        logger.debug("Usage: replay.py <session_id>")
        sys.exit(1)
    replay_session(sys.argv[1])


if __name__ == "__main__":
    main()
