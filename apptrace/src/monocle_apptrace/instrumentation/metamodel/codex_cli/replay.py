"""
Session Replay (transcript-driven)

On Stop, walks the Codex transcript JSONL linearly. Each ``task_started`` →
``task_complete`` pair becomes one turn. Tool calls are detected from
``*_end`` event_msg entries (and from ``function_call_output`` as a fallback for
tools without a structured end event). Subagents are recursively walked from
their own transcript files.

The walk produces fully-populated turn dicts which are then handed to the
ReplayHandler dummy methods — Monocle's wrapper turns each call into a span.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION, SPAN_START_TIME, SPAN_END_TIME
from monocle_apptrace.instrumentation.metamodel.codex_cli._helper import find_subagent_transcript
from monocle_apptrace.instrumentation.metamodel.codex_cli.replay_handlers import ReplayHandler
from monocle_apptrace.instrumentation.metamodel.codex_cli.trace_events import (
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)


_telemetry_ready = False


def _configure_telemetry():
    global _telemetry_ready
    if _telemetry_ready:
        return
    workflow_name = (
        os.environ.get("MONOCLE_WORKFLOW_NAME")
        or os.environ.get("DEFAULT_WORKFLOW_NAME")
        or "codex-cli"
    )
    from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
    setup_monocle_telemetry(workflow_name=workflow_name)
    _telemetry_ready = True


# ── Tool extractors ───────────────────────────────────────────────────────────
#
# Each extractor pulls (tool_name, tool_input, tool_output) from one of Codex's
# *_end event_msg payloads. tool_input/tool_output fall back to whatever was
# captured on the corresponding response_item function_call/custom_tool_call.

def _exec_extractor(payload, pre):
    cmd = payload.get("command") or []
    fallback_input = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    return (
        "exec_command",
        (pre or {}).get("tool_input") or fallback_input,
        payload.get("aggregated_output") or payload.get("stdout") or "",
    )


def _mcp_extractor(payload, pre):
    inv = payload.get("invocation") or {}
    server = inv.get("server", "")
    tool = inv.get("tool", "")
    name = f"mcp__{server}__{tool}" if server else (pre or {}).get("tool_name", tool)
    result = payload.get("result") or {}
    if isinstance(result, dict):
        output = result.get("Ok") or result.get("Err") or result
    else:
        output = result
    return (
        name,
        (pre or {}).get("tool_input") or inv.get("arguments", {}),
        output,
    )


def _patch_extractor(payload, pre):
    return (
        "apply_patch",
        (pre or {}).get("tool_input", ""),
        payload.get("stdout", ""),
    )


def _web_search_extractor(payload, pre):
    action = payload.get("action") or {}
    return (
        "web_search",
        payload.get("query", "") or json.dumps(action),
        "",
    )


def _view_image_extractor(payload, pre):
    return (
        "view_image",
        (pre or {}).get("tool_input") or payload.get("path", ""),
        "",
    )


def _image_gen_extractor(payload, pre):
    return (
        "image_generation",
        (pre or {}).get("tool_input") or payload.get("revised_prompt", ""),
        "[image]",  # base64 result is too large for span attributes
    )


def _collab_verb_extractor(verb_name):
    def _extract(payload, pre):
        status = payload.get("status") or payload.get("agent_statuses")
        return (
            verb_name,
            (pre or {}).get("tool_input", ""),
            json.dumps(status) if status else "",
        )
    return _extract


_TOOL_EXTRACTORS = {
    "exec_command_end": _exec_extractor,
    "mcp_tool_call_end": _mcp_extractor,
    "patch_apply_end": _patch_extractor,
    "web_search_end": _web_search_extractor,
    "view_image_tool_call": _view_image_extractor,
    "image_generation_end": _image_gen_extractor,
    "collab_waiting_end": _collab_verb_extractor("wait_agent"),
    "collab_close_end": _collab_verb_extractor("close_agent"),
    "collab_resume_end": _collab_verb_extractor("resume_agent"),
    "collab_agent_interaction_end": _collab_verb_extractor("send_input"),
}


def _record_tool(turn, pending_calls, ts, payload, extractor):
    cid = payload.get("call_id")
    # Some Codex tools emit two ``*_end`` events per call (e.g. unified_exec
    # startup vs close). Dedupe by call_id so we don't double-span.
    if cid and any(tc.get("call_id") == cid for tc in turn["tool_calls"]):
        return
    pre = pending_calls.pop(cid, None) if cid else None
    name, tool_input, tool_output = extractor(payload, pre)
    turn["tool_calls"].append({
        "tool_name": name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "call_id": cid or "",
        SPAN_START_TIME: (pre or {}).get("start_ts", ts),
        SPAN_END_TIME: ts,
    })


# ── Token binning ─────────────────────────────────────────────────────────────

def _empty_tokens():
    return {"input": 0, "cached": 0, "output": 0, "reasoning": 0}


def _sum_tokens_in_window(token_entries, start_ts, end_ts):
    """Sum ``last_token_usage`` for token_count events whose timestamp falls in
    [start_ts, end_ts]. ISO-8601 timestamps compare lexicographically."""
    acc = _empty_tokens()
    for t in token_entries:
        ts = t.get("ts") or ""
        if start_ts and ts < start_ts:
            continue
        if end_ts and ts > end_ts:
            continue
        acc["input"] += t.get("input_tokens", 0) or 0
        acc["cached"] += t.get("cached_input_tokens", 0) or 0
        acc["output"] += t.get("output_tokens", 0) or 0
        acc["reasoning"] += t.get("reasoning_output_tokens", 0) or 0
    return _format_tokens(acc)


def _format_tokens(acc):
    if not any(acc.values()):
        return {}
    prompt_t = acc["input"] + acc["cached"]
    completion_t = acc["output"] + acc["reasoning"]
    return {
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "total_tokens": prompt_t + completion_t,
        "input_tokens": acc["input"],
        "cache_read_tokens": acc["cached"],
        "reasoning_tokens": acc["reasoning"],
    }


# ── Inference rounds ──────────────────────────────────────────────────────────

def _derive_inference_rounds(turn):
    """Tool boundaries split a turn into inference rounds. Tokens are binned
    into the round whose [start, end] window contains the token_count timestamp.

    No tools  → one round covering the whole turn.
    N tools   → N+1 rounds: [turn_start → tool_1.start], [tool_i.end → tool_{i+1}.start], [tool_N.end → turn_end].
    """
    tool_calls = turn["tool_calls"]
    rounds = []
    turn_start = turn["start_ts"]
    turn_end = turn.get("end_ts") or turn_start
    tokens = turn["tokens"]
    model = turn["model"]
    last_msg = turn.get("last_agent_message", "")

    if not tool_calls:
        rounds.append({
            SPAN_START_TIME: turn_start,
            SPAN_END_TIME: turn_end,
            "model": model,
            "tokens": _sum_tokens_in_window(tokens, turn_start, turn_end),
            "output_text": last_msg,
            "finish_reason": "end_turn",
            "finish_type": "success",
        })
        return rounds

    cursor = turn_start
    for tc in tool_calls:
        round_start = cursor
        round_end = tc[SPAN_START_TIME]
        rounds.append({
            SPAN_START_TIME: round_start,
            SPAN_END_TIME: round_end,
            "model": model,
            "tokens": _sum_tokens_in_window(tokens, round_start, round_end),
            "tool_name": tc["tool_name"],
            "finish_reason": "tool_use",
            "finish_type": "tool_call",
        })
        cursor = tc[SPAN_END_TIME]

    rounds.append({
        SPAN_START_TIME: cursor,
        SPAN_END_TIME: turn_end,
        "model": model,
        "tokens": _sum_tokens_in_window(tokens, cursor, turn_end),
        "output_text": last_msg,
        "finish_reason": "end_turn",
        "finish_type": "success",
    })
    return rounds


# ── Subagent recursion ────────────────────────────────────────────────────────

def _build_subagent(parent_path, thread_id, spawn_payload, spawn_ts):
    sa_path = find_subagent_transcript(parent_path, thread_id)
    if not sa_path:
        return None
    sub_turns, _, sa_model = _walk_turns(str(sa_path), 0, "codex")
    if not sub_turns:
        return None

    sa_start = sub_turns[0]["start_ts"]
    sa_end = sub_turns[-1].get("end_ts") or sa_start
    response = sub_turns[-1].get("last_agent_message", "")

    all_tool_calls = []
    for st in sub_turns:
        all_tool_calls.extend(st.get("tool_calls", []))

    acc = _empty_tokens()
    for st in sub_turns:
        for t in st.get("tokens", []):
            acc["input"] += t.get("input_tokens", 0) or 0
            acc["cached"] += t.get("cached_input_tokens", 0) or 0
            acc["output"] += t.get("output_tokens", 0) or 0
            acc["reasoning"] += t.get("reasoning_output_tokens", 0) or 0

    return {
        "thread_id": thread_id,
        "agent_role": spawn_payload.get("new_agent_role", "agent"),
        "agent_nickname": spawn_payload.get("new_agent_nickname", ""),
        "prompt": spawn_payload.get("prompt", ""),
        "response": response,
        "tool_calls": all_tool_calls,
        "tokens": _format_tokens(acc),
        "model": sa_model or spawn_payload.get("model", "codex"),
        SPAN_START_TIME: sa_start or spawn_ts,
        SPAN_END_TIME: sa_end or spawn_ts,
    }


# ── Transcript walk ───────────────────────────────────────────────────────────

def _walk_turns(transcript_path, start_line, current_model):
    """Walk transcript JSONL from ``start_line`` and yield completed turns.

    Returns (turns, next_start_line, last_seen_model). Partial turns (no
    matching task_complete) are not returned; the cursor stops just after the
    last completed turn so the next replay picks them up.
    """
    if not transcript_path:
        return [], start_line, current_model
    path = Path(transcript_path)
    if not path.exists():
        return [], start_line, current_model

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        logger.debug(f"Error reading transcript {transcript_path}: {e}")
        return [], start_line, current_model

    turns = []
    turn = None
    pending_calls: dict = {}
    next_cursor = start_line

    for offset, line in enumerate(lines[start_line:]):
        idx = start_line + offset
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue

        ts = entry.get("timestamp") or ""
        etype = entry.get("type")
        payload = entry.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        ptype = payload.get("type")

        # Session/turn metadata can update the active model
        if etype == "turn_context":
            m = payload.get("model")
            if m:
                current_model = m
        elif etype == "session_meta":
            mp = payload.get("model_provider")
            if mp and current_model == "codex":
                current_model = mp

        # Turn boundaries
        if etype == "event_msg" and ptype == "task_started":
            turn = {
                "turn_id": payload.get("turn_id", ""),
                "start_ts": ts,
                "model": current_model,
                "user_prompt": "",
                "tool_calls": [],
                "subagents": [],
                "agent_messages": [],
                "last_agent_message": "",
                "tokens": [],
            }
            pending_calls = {}
            continue

        if etype == "event_msg" and ptype == "task_complete":
            if turn is not None:
                turn["end_ts"] = ts
                msg = payload.get("last_agent_message")
                if msg:
                    turn["last_agent_message"] = msg
                turn["duration_ms"] = payload.get("duration_ms")
                turn["time_to_first_token_ms"] = payload.get("time_to_first_token_ms")
                turns.append(turn)
                turn = None
                pending_calls = {}
                next_cursor = idx + 1  # advance cursor past this complete turn
            continue

        if turn is None:
            continue  # outside any turn — ignore

        if etype == "event_msg":
            if ptype == "user_message":
                turn["user_prompt"] = payload.get("message", "") or ""
            elif ptype == "agent_message":
                msg = payload.get("message", "") or ""
                if msg:
                    turn["agent_messages"].append({"ts": ts, "text": msg})
                    turn["last_agent_message"] = msg
            elif ptype == "token_count":
                info = payload.get("info") or {}
                last = info.get("last_token_usage") or {}
                if last:
                    turn["tokens"].append({"ts": ts, **last})
            elif ptype == "collab_agent_spawn_end":
                # Pop the matching function_call so we don't emit a duplicate
                # "spawn_agent" Tool span on top of the Sub-Agent span.
                cid = payload.get("call_id")
                if cid:
                    pending_calls.pop(cid, None)
                sa = _build_subagent(transcript_path, payload.get("new_thread_id"), payload, ts)
                if sa:
                    turn["subagents"].append(sa)
            elif ptype in _TOOL_EXTRACTORS:
                _record_tool(turn, pending_calls, ts, payload, _TOOL_EXTRACTORS[ptype])

        elif etype == "response_item":
            if ptype in ("function_call", "custom_tool_call"):
                cid = payload.get("call_id")
                if cid:
                    pending_calls[cid] = {
                        "tool_name": payload.get("name", ""),
                        "tool_input": payload.get("arguments") or payload.get("input", ""),
                        "start_ts": ts,
                    }
            elif ptype in ("function_call_output", "custom_tool_call_output"):
                # Fallback for tools without a structured *_end event
                # (e.g. update_plan, list_mcp_resources, multi_tool_use.parallel).
                cid = payload.get("call_id")
                pre = pending_calls.pop(cid, None) if cid else None
                if pre:
                    turn["tool_calls"].append({
                        "tool_name": pre.get("tool_name", ""),
                        "tool_input": pre.get("tool_input", ""),
                        "tool_output": payload.get("output", ""),
                        "call_id": cid or "",
                        SPAN_START_TIME: pre.get("start_ts", ts),
                        SPAN_END_TIME: ts,
                    })

    return turns, next_cursor, current_model


# ── Public API ────────────────────────────────────────────────────────────────

def replay_session(session_id: str, transcript_path: Optional[str]) -> None:
    if not session_id or not transcript_path:
        return

    state = load_state(session_id)
    start_line = state.get("transcript_lines_processed", 0)
    current_model = state.get("model", "codex")

    turns, next_cursor, model = _walk_turns(transcript_path, start_line, current_model)
    if not turns:
        return

    _configure_telemetry()
    handler = ReplayHandler()

    for turn in turns:
        rounds = _derive_inference_rounds(turn)
        turn_tokens = _sum_tokens_in_window(turn["tokens"], turn["start_ts"], turn.get("end_ts"))
        handler.handle_turn(
            prompt=turn["user_prompt"],
            response=turn.get("last_agent_message", ""),
            tool_calls=turn["tool_calls"],
            subagents=turn["subagents"],
            inference_rounds=rounds,
            model=turn["model"],
            tokens=turn_tokens,
            turn_id=turn.get("turn_id", ""),
            time_to_first_token_ms=turn.get("time_to_first_token_ms"),
            duration_ms=turn.get("duration_ms"),
            _turn_start=turn["start_ts"],
            _turn_end=turn.get("end_ts"),
            **{
                SPAN_START_TIME: turn["start_ts"],
                SPAN_END_TIME: turn.get("end_ts") or turn["start_ts"],
                AGENT_SESSION: session_id,
            },
        )

    state["transcript_lines_processed"] = next_cursor
    state["model"] = model
    save_state(session_id, state)

    # Force-flush before the hook subprocess exits. Monocle's default
    # BatchSpanProcessor queues spans asynchronously; without this the trace
    # file only appears when a later hook firing happens to drive the flush.
    try:
        from opentelemetry import trace as otel_trace
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
    except Exception as e:
        logger.debug(f"Span flush failed: {e}")
