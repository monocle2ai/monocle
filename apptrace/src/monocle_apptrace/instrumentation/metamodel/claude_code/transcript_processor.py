"""
Claude Code transcript processor.

Parses Claude Code JSONL transcript files and emits OpenTelemetry spans
following the Monocle metamodel pattern with proper entity attributes.

This is invoked by the Claude Code Stop hook (not via monkey-patching,
since Claude Code is a CLI binary, not a Python library).
"""

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from opentelemetry import context as otel_context, trace
from opentelemetry.trace import StatusCode

from monocle_apptrace.instrumentation.metamodel.claude_code._helper import (
    Turn,
    aggregate_usage,
    build_turns,
    classify_tool,
    classify_tool_entity_type,
    extract_text,
    get_content,
    get_message_id,
    get_model,
    get_stop_reason,
    get_timestamp,
    get_usage,
    iter_tool_uses,
    parse_command_skill,
    read_new_jsonl,
    SessionState,
    CLAUDE_CODE_AGENT_TYPE_KEY,
    CLAUDE_CODE_SKILL_TYPE_KEY,
)

logger = logging.getLogger(__name__)

SERVICE_NAME = "claude-cli"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _parse_timestamp_ns(ts: Optional[str]) -> Optional[int]:
    """Convert an ISO8601 timestamp string to nanoseconds since epoch for OTel.

    Returns None if ts is absent or unparseable — callers fall back to OTel's
    default (current wall-clock time), preserving the previous behaviour.
    """
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    except Exception:
        return None


@contextmanager
def _timed_span(
    tracer: trace.Tracer,
    name: str,
    attributes: Dict[str, Any],
    start_ns: Optional[int],
    end_ns: Optional[int],
) -> Generator:
    """Start a span with explicit start/end timestamps while keeping proper OTel context nesting.

    Using tracer.start_span() instead of start_as_current_span() lets us pass
    an end_time to span.end() — necessary because spans are emitted AFTER the
    turn finishes, so we must replay the real event times from the transcript.
    """
    span = tracer.start_span(name=name, start_time=start_ns, attributes=attributes)
    token = otel_context.attach(trace.set_span_in_context(span))
    try:
        yield span
    finally:
        otel_context.detach(token)
        span.end(end_time=end_ns)


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def _build_full_response(turn: Turn) -> str:
    """Build the complete turn response: assistant text + all tool outputs."""
    parts = []
    for assistant_msg in turn.assistant_msgs:
        text = extract_text(get_content(assistant_msg))
        if text:
            parts.append(text)
    for tool_id, tool_output in turn.tool_results_by_id.items():
        if tool_output:
            output_str = tool_output if isinstance(tool_output, str) else json.dumps(tool_output)
            parts.append(output_str)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool description helper
# ---------------------------------------------------------------------------

def _make_tool_description(tool_name: str, tool_input: Any) -> Optional[str]:
    """Generate a concise description for a tool call from its input parameters.

    The NarrativeGraph requires entity.1.description on tool spans to build
    edges. This function derives that description from the tool's input so we
    don't have to store it separately.
    """
    if not isinstance(tool_input, dict):
        return None

    if tool_name == "Bash":
        # Use the explicit description field if provided; otherwise truncate the command
        desc = tool_input.get("description") or tool_input.get("command", "")
        return str(desc)[:120] if desc else None

    if tool_name in ("Read", "Edit", "Write", "NotebookEdit"):
        return tool_input.get("file_path") or None

    if tool_name in ("Grep",):
        pat = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"{pat} in {path}" if path else pat or None

    if tool_name == "Glob":
        return tool_input.get("pattern") or None

    if tool_name in ("WebFetch", "WebSearch"):
        return (tool_input.get("url") or tool_input.get("query") or "")[:120] or None

    if tool_name == "ToolSearch":
        return tool_input.get("query", "")[:80] or None

    if tool_name == "Skill":
        skill = tool_input.get("skill", "")
        args = tool_input.get("args", "")
        return f"{skill}: {args}" if args else skill or None

    if tool_name.startswith("mcp__"):
        # e.g. "mcp__okahu__get_traces" → "okahu / get_traces"
        parts = tool_name.split("__", 2)
        return f"{parts[1]} / {parts[2]}" if len(parts) == 3 else tool_name

    return None


# ---------------------------------------------------------------------------
# Turn emitter
# ---------------------------------------------------------------------------

def _emit_turn(
    tracer: trace.Tracer,
    turn: Turn,
    turn_num: int,
    session_id: str,
    sdk_version: str,
    service_name: str,
    user_name: Optional[str] = None,
) -> bool:
    """Emit spans for one turn: agentic.turn -> agentic.invocation -> per-round inference + tools.

    Structure mirrors Claude's actual execution flow:
      agentic.turn
      └── agentic.invocation  (Claude as the orchestrating agent)
          ├── inference  (round 1: Claude thinks)
          ├── tool/agent (round 1: Claude acts — one span per tool_use)
          ├── inference  (round 2: Claude processes results and thinks again)
          ├── tool/agent (round 2: …)
          └── inference  (final round: Claude generates the answer)

    Each inference span carries that round's own token counts and timing.
    Tool spans are emitted immediately after the inference span that produced them.
    """
    user_text = extract_text(get_content(turn.user_msg))

    if not turn.assistant_msgs:
        return False

    last_assistant = turn.assistant_msgs[-1]
    full_response = _build_full_response(turn)
    model = get_model(turn.assistant_msgs[0])

    turn_start_ns = _parse_timestamp_ns(turn.start_time)
    turn_end_ns = _parse_timestamp_ns(turn.end_time)

    # Stable IDs scoped to this turn — required by NarrativeGraph builder
    turn_id = str(uuid.uuid4())
    invocation_id = str(uuid.uuid4())

    turn_attrs: Dict[str, Any] = {
        "span.type": "agentic.turn",
        "span.subtype": "turn",
        "scope.agentic.session": session_id,
        "scope.agentic.turn": turn_id,
        "turn.number": turn_num,
        "entity.1.type": CLAUDE_CODE_AGENT_TYPE_KEY,
        "entity.1.name": "Claude",
        "workflow.name": service_name,
        "monocle_apptrace.version": sdk_version,
        "monocle.service.name": service_name,
    }
    if user_name:
        turn_attrs["user.name"] = user_name

    with _timed_span(tracer, f"Claude Code - Turn {turn_num}", turn_attrs, turn_start_ns, turn_end_ns) as turn_span:
        turn_span.set_status(StatusCode.OK)
        turn_span.add_event("data.input", {"input": user_text})
        turn_span.add_event("data.output", {"response": full_response})

        # Invocation span wraps all rounds — creates AgentNode + INVOKES edge in NarrativeGraph
        invocation_attrs: Dict[str, Any] = {
            "span.type": "agentic.invocation",
            "scope.agentic.session": session_id,
            "scope.agentic.turn": turn_id,
            "scope.agentic.invocation": invocation_id,
            "entity.1.type": CLAUDE_CODE_AGENT_TYPE_KEY,
            "entity.1.name": "Claude",
            "workflow.name": service_name,
            "monocle_apptrace.version": sdk_version,
        }
        with _timed_span(tracer, "Claude Invocation", invocation_attrs, turn_start_ns, turn_end_ns) as invocation_span:
            invocation_span.set_status(StatusCode.OK)
            invocation_span.add_event("data.input", {"input": user_text})
            invocation_span.add_event("data.output", {"response": full_response})

            # Detect harness-injected skill once per turn (before per-round loop)
            cmd_skill = parse_command_skill(user_text)
            has_explicit_skill = any(
                tu.get("name") == "Skill"
                for am in turn.assistant_msgs
                for tu in iter_tool_uses(get_content(am))
            )
            if cmd_skill and not has_explicit_skill:
                skill_name = cmd_skill["skill_name"]
                skill_input: Dict[str, Any] = {"skill": skill_name}
                if cmd_skill["args"]:
                    skill_input["args"] = cmd_skill["args"]
                if cmd_skill["plugin_name"]:
                    skill_input["plugin"] = cmd_skill["plugin_name"]
                skill_attrs: Dict[str, Any] = {
                    "span.type": "agentic.skill.invocation",
                    "scope.agentic.session": session_id,
                    "scope.agentic.turn": turn_id,
                    "scope.agentic.invocation": invocation_id,
                    "entity.1.type": CLAUDE_CODE_SKILL_TYPE_KEY,
                    "entity.1.name": skill_name,
                    "entity.1.skill_name": skill_name,
                    "entity.1.invocation": "harness",
                    "monocle_apptrace.version": sdk_version,
                    "workflow.name": service_name,
                }
                if cmd_skill["args"]:
                    skill_attrs["entity.1.skill_args"] = cmd_skill["args"]
                if cmd_skill["plugin_name"]:
                    skill_attrs["entity.1.plugin_name"] = cmd_skill["plugin_name"]
                with _timed_span(tracer, f"Skill: {skill_name}", skill_attrs, turn_start_ns, turn_end_ns) as skill_span:
                    skill_span.set_status(StatusCode.OK)
                    skill_span.add_event("data.input", {"input": json.dumps(skill_input)})
                    skill_span.add_event("data.output", {"response": f"/{cmd_skill['command_name']}"})

            # ---------------------------------------------------------------
            # Per-round loop: one inference span + tool spans per LLM round.
            # This faithfully mirrors Claude's actual execution:
            #   inference → tool calls → (results arrive) → inference → …
            # ---------------------------------------------------------------
            total_tool_spans = 0
            num_rounds = len(turn.assistant_msgs)
            for i, assistant_msg in enumerate(turn.assistant_msgs):
                round_usage = get_usage(assistant_msg)
                round_stop_reason = get_stop_reason(assistant_msg) or "end_turn"
                round_finish_type = "tool_call" if round_stop_reason == "tool_use" else "success"
                round_text = extract_text(get_content(assistant_msg))
                round_msg_id = get_message_id(assistant_msg) or ""

                # Inference span timing:
                #   start = user message time (round 0) or when all previous tools finished
                #   end   = this assistant message timestamp (when LLM finished generating)
                round_inf_end_ns = _parse_timestamp_ns(get_timestamp(assistant_msg))
                if i == 0:
                    round_inf_start_ns = turn_start_ns
                else:
                    prev_tool_ids = [
                        tu.get("id") for tu in iter_tool_uses(get_content(turn.assistant_msgs[i - 1]))
                    ]
                    prev_result_times = [
                        _parse_timestamp_ns(turn.tool_result_times_by_id.get(tid))
                        for tid in prev_tool_ids if tid
                    ]
                    prev_result_times = [t for t in prev_result_times if t]
                    round_inf_start_ns = (
                        max(prev_result_times) if prev_result_times
                        else _parse_timestamp_ns(get_timestamp(turn.assistant_msgs[i - 1]))
                    )

                # Per-round token counts from this LLM call.
                output_t = round_usage.get("output_tokens") or 0
                is_final_round = (i == num_rounds - 1)
                round_metadata: Dict[str, Any] = {
                    "finish_reason": round_stop_reason,
                    "finish_type": round_finish_type,
                }
                if output_t:
                    round_metadata["completion_tokens"] = output_t
                if is_final_round:
                    # Aggregate token counts across ALL rounds so prompt_tokens and
                    # total_tokens reflect the full turn, not just the last LLM call.
                    total_usage = aggregate_usage(turn.assistant_msgs)
                    total_input_t = total_usage.get("input_tokens") or 0
                    total_cache_read_t = total_usage.get("cache_read_tokens") or 0
                    total_cache_creation_t = total_usage.get("cache_creation_tokens") or 0
                    total_output_t = total_usage.get("output_tokens") or 0
                    total_prompt_t = total_input_t + total_cache_read_t + total_cache_creation_t
                    if total_prompt_t:
                        round_metadata["prompt_tokens"] = total_prompt_t
                    if total_prompt_t or total_output_t:
                        round_metadata["total_tokens"] = total_prompt_t + total_output_t
                    if total_cache_read_t:
                        round_metadata["cache_read_tokens"] = total_cache_read_t
                    if total_cache_creation_t:
                        round_metadata["cache_creation_tokens"] = total_cache_creation_t

                inf_name = (
                    "Claude Inference" if num_rounds == 1
                    else f"Claude Inference ({i + 1}/{num_rounds})"
                )
                round_inf_attrs: Dict[str, Any] = {
                    "span.type": "inference",
                    "scope.agentic.session": session_id,
                    "scope.agentic.turn": turn_id,
                    "scope.agentic.invocation": invocation_id,
                    "entity.1.type": "inference.anthropic",
                    "entity.1.provider_name": "anthropic",
                    "entity.2.name": model,
                    "entity.2.type": f"model.llm.{model}",
                    "gen_ai.system": "anthropic",
                    "gen_ai.request.model": model,
                    "gen_ai.response.id": round_msg_id,
                    "monocle_apptrace.version": sdk_version,
                    "workflow.name": service_name,
                }
                # If Claude produced no text in this round (pure tool dispatch),
                # summarise what it decided to do so the span output isn't empty.
                if not round_text:
                    tool_uses_here = iter_tool_uses(get_content(assistant_msg))
                    dispatched = []
                    for tu in tool_uses_here:
                        name = tu.get("name", "unknown")
                        if name == "Agent":
                            subtype = tu.get("input", {}).get("subagent_type", "agent")
                            dispatched.append(f"Agent({subtype})")
                        else:
                            dispatched.append(name)
                    round_text = f"[Dispatched: {', '.join(dispatched)}]" if dispatched else "[tool dispatch]"

                with _timed_span(tracer, inf_name, round_inf_attrs, round_inf_start_ns, round_inf_end_ns) as inf_span:
                    inf_span.set_status(StatusCode.OK)
                    inf_span.add_event("data.input", {"input": user_text})
                    inf_span.add_event("data.output", {"response": round_text})
                    inf_span.add_event("metadata", round_metadata)

                # Tool/agent spans for this round — emitted right after the inference
                # that triggered them. Parallel tools share the same start time (when
                # the LLM finished) and each ends when its own result arrived.
                tool_call_start_ns = round_inf_end_ns
                for tool_use in iter_tool_uses(get_content(assistant_msg)):
                    tool_id = tool_use.get("id", "")
                    tool_name = tool_use.get("name", "unknown")
                    tool_input = tool_use.get("input", {})
                    tool_output = turn.tool_results_by_id.get(tool_id, "")

                    tool_result_ts = turn.tool_result_times_by_id.get(tool_id)
                    tool_call_end_ns = _parse_timestamp_ns(tool_result_ts) or turn_end_ns

                    span_type = classify_tool(tool_name)
                    entity_type = classify_tool_entity_type(tool_name)

                    input_str = json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
                    output_str = (
                        tool_output if isinstance(tool_output, str)
                        else json.dumps(tool_output) if tool_output
                        else ""
                    )

                    tool_description = _make_tool_description(tool_name, tool_input)
                    span_attrs: Dict[str, Any] = {
                        "span.type": span_type,
                        "scope.agentic.session": session_id,
                        "scope.agentic.turn": turn_id,
                        "scope.agentic.invocation": invocation_id,
                        "entity.1.type": entity_type,
                        "entity.1.name": tool_name,
                        "monocle_apptrace.version": sdk_version,
                        "workflow.name": service_name,
                    }
                    if tool_description:
                        span_attrs["entity.1.description"] = tool_description
                    span_name = f"Tool: {tool_name}"
                    if tool_name == "Agent" and isinstance(tool_input, dict):
                        subagent_type = tool_input.get("subagent_type", "general-purpose")
                        span_attrs["entity.1.name"] = subagent_type
                        span_attrs["entity.1.description"] = tool_input.get("description", "")
                        # Link subagent back to the delegating Claude invocation span so
                        # the NarrativeGraph builder can create a DELEGATES_TO edge.
                        # Also give this subagent its own invocation_id so the graph
                        # builder does not re-process Claude's tool spans under this anchor.
                        span_attrs["entity.1.from_agent"] = "Claude"
                        span_attrs["entity.1.from_agent_span_id"] = format(
                            invocation_span.get_span_context().span_id, "016x"
                        )
                        span_attrs["scope.agentic.invocation"] = str(uuid.uuid4())
                    elif tool_name == "Skill" and isinstance(tool_input, dict):
                        skill_nm = tool_input.get("skill", "unknown")
                        span_attrs["entity.1.name"] = skill_nm
                        span_attrs["entity.1.skill_name"] = skill_nm
                        if tool_input.get("args"):
                            span_attrs["entity.1.skill_args"] = tool_input.get("args")
                        span_name = f"Skill: {skill_nm}"

                    with _timed_span(tracer, span_name, span_attrs, tool_call_start_ns, tool_call_end_ns) as tool_span:
                        tool_span.set_status(StatusCode.OK)
                        tool_span.add_event("data.input", {"input": input_str})
                        tool_span.add_event("data.output", {"response": output_str})

                    total_tool_spans += 1

            logger.debug(
                "turn %d: %d LLM rounds, %d tool spans",
                turn_num, num_rounds, total_tool_spans,
            )

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_transcript(
    session_id: str,
    turns: List[Turn],
    tracer: trace.Tracer,
    sdk_version: str,
    service_name: str = SERVICE_NAME,
    start_turn: int = 0,
    user_name: Optional[str] = None,
) -> int:
    """Emit Monocle-compatible spans for a list of turns.

    Creates a workflow root span wrapping all turn spans.
    Okahu requires this workflow span for span detail retrieval.

    Returns the number of turns emitted.
    """
    if not turns:
        return 0

    emitted = 0

    # Workflow timing: first turn start → last turn end
    workflow_start_ns = _parse_timestamp_ns(turns[0].start_time)
    workflow_end_ns = _parse_timestamp_ns(turns[-1].end_time)

    workflow_attrs: Dict[str, Any] = {
        "span.type": "workflow",
        "scope.agentic.session": session_id,
        "entity.1.name": service_name,
        "entity.1.type": "workflow.claude_code",
        "entity.2.type": "app_hosting.generic",
        "entity.2.name": "generic",
        "monocle_apptrace.version": sdk_version,
        "monocle_apptrace.language": "python",
        "workflow.name": service_name,
    }
    if user_name:
        workflow_attrs["user.name"] = user_name

    with _timed_span(tracer, "workflow", workflow_attrs, workflow_start_ns, workflow_end_ns) as workflow_span:
        workflow_span.set_status(StatusCode.OK)
        for i, turn in enumerate(turns):
            turn_num = start_turn + i + 1
            if _emit_turn(tracer, turn, turn_num, session_id, sdk_version, service_name, user_name):
                emitted += 1

    return emitted


def process_transcript_file(
    session_id: str,
    transcript_path: Path,
    tracer: trace.Tracer,
    sdk_version: str,
    service_name: str = SERVICE_NAME,
    session_state: Optional[SessionState] = None,
    user_name: Optional[str] = None,
) -> tuple:
    """Higher-level API: read new JSONL from a transcript file, build turns, emit spans.

    Returns (emitted_count, updated_session_state).
    """
    if session_state is None:
        session_state = SessionState()

    msgs, session_state = read_new_jsonl(transcript_path, session_state)
    if not msgs:
        return 0, session_state

    turns = build_turns(msgs)
    if not turns:
        return 0, session_state

    emitted = process_transcript(
        session_id=session_id,
        turns=turns,
        tracer=tracer,
        sdk_version=sdk_version,
        service_name=service_name,
        start_turn=session_state.turn_count,
        user_name=user_name,
    )
    session_state.turn_count += len(turns)

    return emitted, session_state
