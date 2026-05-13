"""Walk ~/.copilot/session-state/<session_id>/events.jsonl and emit one span tree per turn.

Copilot CLI writes this file natively — each interaction (user prompt → agent stop)
is grouped by interactionId. We trigger replay from the Stop hook.
"""

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

try:
    import fcntl  # POSIX only; absent on Windows
except ImportError:
    fcntl = None

from monocle_apptrace.instrumentation.common.constants import (
    AGENT_SESSION,
    COPILOT_TURN_SCOPE,
    SPAN_START_TIME,
    SPAN_END_TIME,
)
from monocle_apptrace.instrumentation.metamodel.copilot_cli.replay_handlers import (
    InterruptedTurnError,
    ReplayHandler,
)
from monocle_apptrace.instrumentation.metamodel.copilot_cli.trace_events import (
    _sessions_dir,
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)

# Copilot CLI internal tools observed in events.jsonl that aren't user-visible.
# These produce no useful tool span; skip their pre/complete pairs entirely.
_INTERNAL_TOOLS = {"report_intent", "fetch_copilot_cli_documentation", "ask_user"}

_telemetry_ready = False


@contextmanager
def _session_lock(session_id: str):
    """Advisory file lock per session_id so two concurrent Stop hooks don't
    both replay the same events. Lock auto-releases when the holding process
    dies. No-op on platforms without fcntl (Windows)."""
    sessions = _sessions_dir()
    sessions.mkdir(parents=True, exist_ok=True)
    lock_path = sessions / f".{session_id}.lock"
    lock_f = lock_path.open("w")
    try:
        if fcntl is not None:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            except OSError as e:
                logger.debug("flock unavailable, proceeding unlocked: %s", e)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_f.close()


def _configure_telemetry():
    global _telemetry_ready
    if _telemetry_ready:
        return
    from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value
    from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
    workflow_name = get_monocle_env_value("MONOCLE_WORKFLOW_NAME") or Path.cwd().name
    setup_monocle_telemetry(workflow_name=workflow_name)
    _telemetry_ready = True


def _events_path(session_id: str) -> Optional[Path]:
    p = Path.home() / ".copilot" / "session-state" / session_id / "events.jsonl"
    return p if p.exists() else None


def _load_events(session_id: str) -> list:
    p = _events_path(session_id)
    if not p:
        return []
    events = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def _build_inference_round(interaction: dict, prompt: str) -> dict:
    """ONE inference round per interaction.

    Copilot's per-message rounds (turnIds) are LLM continuation calls of the
    same logical inference — not separate user-visible reasoning steps. We sum
    outputTokens across the whole interaction (the only token field available
    at message granularity).

    Session-level prompt/cache token totals live on `session.shutdown` and are
    surfaced separately; they aren't attributable to a single interaction.
    """
    assistant_msgs = interaction.get("assistant_messages", [])
    if not assistant_msgs:
        return {}

    first_msg = assistant_msgs[0]
    last_msg = assistant_msgs[-1]

    round_start = interaction.get("turn_start") or first_msg.get("round_start") or first_msg["timestamp"]
    round_end = last_msg["timestamp"]

    final_content = next((m.get("content", "") for m in reversed(assistant_msgs) if m.get("content")), "")
    completion_tokens = sum(m.get("output_tokens", 0) or 0 for m in assistant_msgs)

    dispatched_tools = []
    for m in assistant_msgs:
        for r in m.get("tool_requests") or []:
            name = r.get("name")
            if name and name not in _INTERNAL_TOOLS:
                dispatched_tools.append(name)

    finish_reason = "tool_use" if dispatched_tools else "end_turn"
    finish_type = "tool_call" if dispatched_tools else "success"

    return {
        SPAN_START_TIME: round_start,
        SPAN_END_TIME: round_end,
        "model": last_msg.get("model") or first_msg.get("model", "copilot"),
        "tokens": {"completion_tokens": completion_tokens} if completion_tokens else {},
        "input_text": prompt,
        "output_text": final_content,
        "finish_reason": finish_reason,
        "finish_type": finish_type,
    }


def _walk_interactions(events: list) -> tuple:
    """Parse events.jsonl into completed interactions.

    Returns (interactions_list, model).
    """
    interactions = {}        # interactionId → dict
    current_iid = None
    pending_round_start = {} # interactionId → pending turn_start timestamp
    pending_tools = {}       # toolCallId → {tool_name, arguments, start_ts, iid}
    model = "copilot"

    for event in events:
        etype = event.get("type", "")
        data = event.get("data") or {}
        ts = event.get("timestamp", "")

        if etype == "session.model_change":
            if data.get("newModel"):
                model = data["newModel"]
            continue

        if etype == "user.message":
            iid = data.get("interactionId", "")
            if not iid:
                continue
            current_iid = iid
            interactions[iid] = {
                "interaction_id": iid,
                "prompt": data.get("content", ""),
                "assistant_messages": [],
                "tool_calls": [],
                "turn_start": ts,   # user.message timestamp = interaction start
                "turn_end": ts,     # will be updated by each assistant.turn_end
            }

        elif etype == "assistant.turn_start":
            # Fires once per inference round — store as the pending round start
            iid = data.get("interactionId", "") or current_iid
            if iid and iid in interactions:
                pending_round_start[iid] = ts

        elif etype == "assistant.message":
            iid = data.get("interactionId", "") or current_iid
            if iid and iid in interactions:
                round_start = pending_round_start.pop(iid, interactions[iid]["turn_start"])
                tool_reqs = data.get("toolRequests") or []
                interactions[iid]["assistant_messages"].append({
                    "round_start": round_start,
                    "timestamp": ts,
                    "content": data.get("content", "") or "",
                    "tool_requests": tool_reqs,
                    "output_tokens": data.get("outputTokens", 0) or 0,
                    "model": model,
                })

        elif etype == "tool.execution_start":
            call_id = data.get("toolCallId", "")
            tool_name = data.get("toolName", "")
            if call_id and tool_name not in _INTERNAL_TOOLS:
                pending_tools[call_id] = {
                    "tool_name": tool_name,
                    "arguments": data.get("arguments", {}),
                    "start_ts": ts,
                    "iid": current_iid,
                }

        elif etype == "tool.execution_complete":
            call_id = data.get("toolCallId", "")
            pre = pending_tools.pop(call_id, None)
            if pre:
                target_iid = data.get("interactionId") or pre.get("iid") or current_iid
                if target_iid and target_iid in interactions:
                    result = data.get("result") or {}
                    error = data.get("error") or {}
                    success = data.get("success", True)
                    interactions[target_iid]["tool_calls"].append({
                        "tool_name": pre["tool_name"],
                        "tool_input": pre["arguments"],
                        "tool_output": result.get("content", "") if success else "",
                        "call_id": call_id,
                        "failed": not success,
                        "error_message": error.get("message", "") if not success else "",
                        "error_code": error.get("code", "") if not success else "",
                        SPAN_START_TIME: pre["start_ts"],
                        SPAN_END_TIME: ts,
                    })

        elif etype == "assistant.turn_end":
            # No interactionId on this event — use current_iid
            if current_iid and current_iid in interactions:
                interactions[current_iid]["turn_end"] = ts

    # Surface tool starts that never got a matching complete as failed
    # tool spans. These are typically interaction interrupts (Ctrl+C, network
    # drop). Without this they'd silently disappear from traces.
    for call_id, pre in pending_tools.items():
        iid = pre.get("iid")
        if iid and iid in interactions:
            interactions[iid]["tool_calls"].append({
                "tool_name": pre["tool_name"],
                "tool_input": pre["arguments"],
                "tool_output": "",
                "call_id": call_id,
                "failed": True,
                "error_message": "tool execution never completed",
                "error_code": "interrupted",
                SPAN_START_TIME: pre["start_ts"],
                SPAN_END_TIME: pre["start_ts"],
            })

    # Emit every interaction that has a user.message. Interactions without a
    # following assistant.message are marked interrupted; handle_invocation
    # raises so their turn+invocation spans get ERROR status. Empty prompts
    # flow through normally (the user explicitly submitted a blank line).
    completed = []
    for itr in interactions.values():
        if not itr.get("assistant_messages"):
            itr["interrupted"] = True
        completed.append(itr)
    completed.sort(key=lambda x: x.get("turn_start") or "")
    return completed, model


def _extract_session_totals(events: list) -> dict:
    """Find the latest session.shutdown row and return per-model totals + perf metrics.

    Returns {} if no shutdown row is present yet (typical — shutdown writes
    after our last Stop hook fires). Caller decides where to surface them.
    """
    shutdown = next((e for e in reversed(events) if e.get("type") == "session.shutdown"), None)
    if not shutdown:
        return {}
    data = shutdown.get("data") or {}
    metrics = data.get("modelMetrics") or {}
    # Aggregate across models in case multi-model session
    usage_acc = {"inputTokens": 0, "outputTokens": 0, "cacheReadTokens": 0,
                 "cacheWriteTokens": 0, "reasoningTokens": 0}
    request_count = 0
    for _, m in metrics.items():
        usage = m.get("usage") or {}
        for k in usage_acc:
            usage_acc[k] += usage.get(k, 0) or 0
        request_count += (m.get("requests") or {}).get("count", 0) or 0

    return {
        "prompt_tokens": usage_acc["inputTokens"],
        "completion_tokens": usage_acc["outputTokens"],
        "cache_read_tokens": usage_acc["cacheReadTokens"],
        "cache_creation_tokens": usage_acc["cacheWriteTokens"],
        "reasoning_tokens": usage_acc["reasoningTokens"],
        "total_tokens": usage_acc["inputTokens"] + usage_acc["outputTokens"],
        "request_count": request_count,
        "total_api_duration_ms": data.get("totalApiDurationMs", 0) or 0,
        "shutdown_ts": shutdown.get("timestamp", ""),
    }


def replay_session(session_id: str) -> None:
    """Replay a Copilot CLI session's events.jsonl into Monocle spans.

    Protected by a per-session advisory file lock so two concurrent Stop hooks
    (a real race during fast multi-turn sessions) don't double-emit spans.
    """
    if not session_id:
        return
    with _session_lock(session_id):
        _replay_session_locked(session_id)


def _replay_session_locked(session_id: str) -> None:
    events = _load_events(session_id)
    if not events:
        return

    state = load_state(session_id)
    already_processed = set(state.get("interactions_processed", []))
    shutdown_emitted = state.get("shutdown_emitted", False)

    interactions, model = _walk_interactions(events)
    new_interactions = [i for i in interactions if i["interaction_id"] not in already_processed]
    session_totals = _extract_session_totals(events) if not shutdown_emitted else {}

    if not new_interactions and not session_totals:
        return

    _configure_telemetry()
    handler = ReplayHandler()

    for interaction in new_interactions:
        iid = interaction["interaction_id"]
        prompt = interaction["prompt"]
        turn_start = interaction.get("turn_start", "")
        turn_end = interaction.get("turn_end", turn_start)
        interrupted = bool(interaction.get("interrupted"))

        assistant_msgs = interaction.get("assistant_messages", [])
        response = next((m["content"] for m in reversed(assistant_msgs) if m.get("content")), "")

        round_dict = _build_inference_round(interaction, prompt)
        inference_rounds = [round_dict] if round_dict else []

        try:
            handler.handle_turn(
                prompt=prompt,
                response=response,
                tool_calls=interaction["tool_calls"],
                inference_rounds=inference_rounds,
                model=model,
                interrupted=interrupted,
                _turn_start=turn_start,
                _turn_end=turn_end,
                **{
                    SPAN_START_TIME: turn_start,
                    SPAN_END_TIME: turn_end,
                    AGENT_SESSION: session_id,
                    COPILOT_TURN_SCOPE: iid,
                },
            )
        except InterruptedTurnError:
            # Span already recorded ERROR status via the wrapper's exception flow.
            pass
        already_processed.add(iid)

    # If session.shutdown appeared, surface its totals on a session-scoped span.
    # Time bounds span the WHOLE session: first interaction's turn_start (=
    # earliest user.message) → shutdown_ts. Falls back to shutdown_ts on both
    # ends if no interactions exist (degenerate case).
    if session_totals:
        shutdown_ts = session_totals.get("shutdown_ts", "")
        session_start = interactions[0]["turn_start"] if interactions else shutdown_ts
        session_end = shutdown_ts or session_start
        if session_start and session_end:
            handler.handle_session_summary(
                totals=session_totals,
                model=model,
                **{
                    SPAN_START_TIME: session_start,
                    SPAN_END_TIME: session_end,
                    AGENT_SESSION: session_id,
                },
            )
            shutdown_emitted = True

    state["interactions_processed"] = list(already_processed)
    state["model"] = model
    state["shutdown_emitted"] = shutdown_emitted
    save_state(session_id, state)
    logger.debug(
        "Copilot replay: %d new interactions, session_totals=%s, session=%s",
        len(new_interactions), bool(session_totals), session_id,
    )
