__all__ = [
    "extract_conversation_id",
    "get_agent_name",
    "get_conversation_agent_name",
    "extract_turn_input",
    "extract_turn_output",
    "extract_step_input",
    "extract_step_output",
    "get_tool_name",
    "get_source_agent",
    "extract_tool_input",
    "extract_tool_response",
]

from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES


def _content_to_text(content):
    if content is None:
        return None
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else None


def _get_events(instance):
    # Newest-first, lazily: EventLog indexing can hit disk per event on
    # persisted conversations, and every caller stops at the first match.
    state = getattr(instance, "state", None) or getattr(instance, "_state", None)
    events = getattr(state, "events", None)
    if events is None:
        return
    for i in range(len(events) - 1, -1, -1):
        yield events[i]


def _event_output_text(event):
    kind = type(event).__name__
    if kind == "MessageEvent" and getattr(event, "source", None) == "agent":
        return _content_to_text(event.llm_message.content)
    if kind == "ActionEvent":
        # FinishTool carries the final answer on its action's `message` field
        message = getattr(getattr(event, "action", None), "message", None)
        return message or getattr(event, "summary", None)
    if kind == "AgentErrorEvent":
        return event.error
    return None


def extract_conversation_id(instance):
    state = getattr(instance, "state", None) or getattr(instance, "_state", None)
    conversation_id = getattr(state, "id", None)
    return str(conversation_id) if conversation_id else None


def get_agent_name(instance):
    return getattr(instance, "name", None)


def get_conversation_agent_name(instance):
    return getattr(getattr(instance, "agent", None), "name", None)


def extract_turn_input(arguments):
    for event in _get_events(arguments["instance"]):
        if type(event).__name__ == "MessageEvent" and getattr(event, "source", None) == "user":
            return _content_to_text(event.llm_message.content)
    return None


def extract_turn_output(arguments):
    # Prefer a definitive outcome (agent message, finish message, error); fall
    # back to the newest action summary only when nothing better exists (e.g.
    # the run stopped on max-iterations mid-action).
    fallback = None
    for event in _get_events(arguments["instance"]):
        kind = type(event).__name__
        if kind == "MessageEvent" and getattr(event, "source", None) == "agent":
            return _content_to_text(event.llm_message.content)
        if kind == "AgentErrorEvent":
            return event.error
        if kind == "ActionEvent":
            message = getattr(getattr(event, "action", None), "message", None)
            if message:
                return message
            if fallback is None:
                fallback = getattr(event, "summary", None)
    return fallback


def _get_step_conversation(arguments):
    if arguments["args"]:
        return arguments["args"][0]
    return arguments["kwargs"].get("conversation")


def extract_step_input(arguments):
    conversation = _get_step_conversation(arguments)
    if conversation is None:
        return None
    for event in _get_events(conversation):
        kind = type(event).__name__
        if kind == "MessageEvent" and getattr(event, "source", None) == "user":
            return _content_to_text(event.llm_message.content)
        if kind == "ObservationEvent":
            return _content_to_text(getattr(event.observation, "to_llm_content", None))
    return None


def extract_step_output(arguments):
    conversation = _get_step_conversation(arguments)
    if conversation is None:
        return None
    for event in _get_events(conversation):
        text = _event_output_text(event)
        if text:
            return text
    return None


def _get_action_event(arguments):
    if arguments["args"]:
        return arguments["args"][0]
    return arguments["kwargs"].get("action")


def get_source_agent(arguments):
    parent_span = arguments.get("parent_span")
    if parent_span and parent_span.attributes.get("span.type") == SPAN_TYPES.AGENTIC_INVOCATION:
        return parent_span.attributes.get("entity.1.name")
    return None


def get_tool_name(arguments):
    action_event = _get_action_event(arguments)
    return getattr(action_event, "tool_name", None)


def extract_tool_input(arguments):
    action_event = _get_action_event(arguments)
    action = getattr(action_event, "action", None)
    if action is not None:
        return action.model_dump_json(exclude_none=True)
    tool_call = getattr(action_event, "tool_call", None)
    return getattr(tool_call, "arguments", None)


def extract_tool_response(result):
    if not result:
        return None
    event = result[0]
    kind = type(event).__name__
    if kind == "ObservationEvent":
        return _content_to_text(getattr(event.observation, "to_llm_content", None))
    if kind == "AgentErrorEvent":
        return event.error
    if kind == "UserRejectObservation":
        return event.rejection_reason
    return None


