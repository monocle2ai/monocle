"""On Stop, walks the Codex transcript JSONL and emits one span tree per turn."""

import json
import logging
import os
import re
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


# Universal signals: response_item *_call (paired by call_id with *_call_output)
# for tools, event_msg collab_agent_spawn_end for subagents. No per-tool dispatch.


_ERROR_PREFIXES = ("unable to", "error:", "failed:", "no such", "denied",
                   "invalid", "not found", "cannot ")
_EXEC_FAILURE_RE = re.compile(r"Process exited with code (\d+)")


def _output_signals_failure(output):
    """Fragile to Codex's output format — accepted to avoid a per-tool table."""
    if not isinstance(output, str) or not output:
        return False
    head = output.strip().lower()[:120]
    if any(head.startswith(p) for p in _ERROR_PREFIXES):
        return True
    m = _EXEC_FAILURE_RE.search(output)
    if m and int(m.group(1)) != 0:
        return True
    if '"Err"' in output or '"isError":true' in output:
        return True
    return False


# ── Token binning ─────────────────────────────────────────────────────────────

def _empty_tokens():
    return {"input": 0, "cached": 0, "output": 0, "reasoning": 0}


def _sum_tokens_in_window(token_entries, start_ts, end_ts):
    """Sum token_count entries whose timestamp falls in [start_ts, end_ts]."""
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
    """N tool calls produce N+1 rounds; tokens get binned by timestamp."""
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

    all_tools = [tc for st in sub_turns for tc in st.get("tool_calls", [])]
    all_tokens = [t for st in sub_turns for t in st.get("tokens", [])]

    return {
        "thread_id": thread_id,
        "agent_role": spawn_payload.get("new_agent_role", "agent"),
        "agent_nickname": spawn_payload.get("new_agent_nickname", ""),
        "prompt": spawn_payload.get("prompt", ""),
        "response": response,
        "tool_calls": all_tools,
        "tokens": _sum_tokens_in_window(all_tokens, None, None),
        "model": sa_model or spawn_payload.get("model", "codex"),
        SPAN_START_TIME: sa_start or spawn_ts,
        SPAN_END_TIME: sa_end or spawn_ts,
    }


# ── Transcript walk ───────────────────────────────────────────────────────────

def _walk_turns(transcript_path, start_line, current_model):
    """Walk transcript from start_line. Returns (completed_turns, next_cursor, model)."""
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
    handled_subagent_calls: set = set()
    next_cursor = start_line

    def _flush_inline_calls():
        """Emit any *_call entries that never got a matching *_call_output."""
        for cid, pre in pending_calls.items():
            if cid in handled_subagent_calls:
                continue
            output = pre.get("inline_output", "")
            turn["tool_calls"].append({
                "tool_name": pre["tool_name"],
                "tool_input": pre["tool_input"],
                "tool_output": output,
                "call_id": cid,
                "failed": _output_signals_failure(output),
                SPAN_START_TIME: pre["start_ts"],
                SPAN_END_TIME: pre["start_ts"],
            })

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

        if etype == "turn_context":
            m = payload.get("model")
            if m:
                current_model = m
        elif etype == "session_meta":
            mp = payload.get("model_provider")
            if mp and current_model == "codex":
                current_model = mp

        if etype == "event_msg" and ptype == "task_started":
            turn = {
                "turn_id": payload.get("turn_id", ""),
                "start_ts": ts,
                "model": current_model,
                "user_prompt": "",
                "tool_calls": [],
                "subagents": [],
                "last_agent_message": "",
                "tokens": [],
            }
            pending_calls = {}
            handled_subagent_calls = set()
            continue

        if etype == "event_msg" and ptype == "task_complete":
            if turn is not None:
                _flush_inline_calls()
                turn["end_ts"] = ts
                msg = payload.get("last_agent_message")
                if msg:
                    turn["last_agent_message"] = msg
                turn["duration_ms"] = payload.get("duration_ms")
                turn["time_to_first_token_ms"] = payload.get("time_to_first_token_ms")
                turns.append(turn)
                turn = None
                pending_calls = {}
                handled_subagent_calls = set()
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
                    turn["last_agent_message"] = msg
            elif ptype == "token_count":
                info = payload.get("info") or {}
                last = info.get("last_token_usage") or {}
                if last:
                    turn["tokens"].append({"ts": ts, **last})
            elif ptype == "collab_agent_spawn_end":
                # Suppress the matching function_call_output — it'd duplicate the Sub-Agent.
                cid = payload.get("call_id")
                if cid:
                    handled_subagent_calls.add(cid)
                    pending_calls.pop(cid, None)
                sa = _build_subagent(transcript_path, payload.get("new_thread_id"), payload, ts)
                if sa:
                    turn["subagents"].append(sa)

        elif etype == "response_item" and isinstance(ptype, str):
            if ptype.endswith("_call_output"):
                cid = payload.get("call_id")
                if cid in handled_subagent_calls:
                    continue  # spawn_agent — already a Sub-Agent span
                pre = pending_calls.pop(cid, None) if cid else None
                if pre:
                    output = payload.get("output", "")
                    turn["tool_calls"].append({
                        "tool_name": pre["tool_name"],
                        "tool_input": pre["tool_input"],
                        "tool_output": output,
                        "call_id": cid or "",
                        "failed": _output_signals_failure(output),
                        SPAN_START_TIME: pre["start_ts"],
                        SPAN_END_TIME: ts,
                    })
            elif ptype.endswith("_call"):
                cid = payload.get("call_id") or payload.get("id") or ""
                tool_name = payload.get("name") or ptype[: -len("_call")]
                tool_input = (
                    payload.get("arguments")
                    or payload.get("input")
                    or payload.get("revised_prompt")
                    or payload.get("query")
                    or json.dumps(payload.get("action")) if payload.get("action") else ""
                )
                inline_output = payload.get("result") or ""
                if isinstance(inline_output, str) and len(inline_output) > 1000:
                    inline_output = "[truncated]"
                if cid:
                    pending_calls[cid] = {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "inline_output": inline_output,
                        "start_ts": ts,
                    }
                else:
                    # No id (web_search_call) — never pairable, emit now.
                    turn["tool_calls"].append({
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_output": inline_output,
                        "call_id": "",
                        "failed": _output_signals_failure(inline_output),
                        SPAN_START_TIME: ts,
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
