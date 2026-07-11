from opentelemetry import context as otel_context
from opentelemetry.trace import INVALID_SPAN, NonRecordingSpan, SpanContext, TraceFlags

from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION, SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import (
    get_scopes,
    remove_scope,
    set_monocle_span_in_context,
)
from monocle_apptrace.instrumentation.metamodel.openhands import _helper
from monocle_apptrace.instrumentation.metamodel.openhands.openhands_processor import (
    OpenHandsSpanHandler,
    OpenHandsToolHandler,
)


class TextContent:
    def __init__(self, text):
        self.text = text


class Message:
    def __init__(self, text):
        self.content = [TextContent(text)]


class MessageEvent:
    def __init__(self, source, text):
        self.source = source
        self.llm_message = Message(text)


class FinishAction:
    def __init__(self, message):
        self.message = message

    def model_dump_json(self, **kwargs):
        return '{"message": "%s"}' % self.message


class ActionEvent:
    def __init__(self, tool_name, action=None, summary=None):
        self.source = "agent"
        self.tool_name = tool_name
        self.action = action
        self.summary = summary


class Observation:
    def __init__(self, text):
        self.to_llm_content = [TextContent(text)]


class ObservationEvent:
    def __init__(self, tool_name, text):
        self.source = "environment"
        self.tool_name = tool_name
        self.observation = Observation(text)


class AgentErrorEvent:
    def __init__(self, error):
        self.source = "agent"
        self.error = error


class State:
    def __init__(self, events, conversation_id="conv-123"):
        self.id = conversation_id
        self.events = events


class Conversation:
    def __init__(self, events, conversation_id="conv-123"):
        self.state = State(events, conversation_id)
        self.agent = type("AgentStub", (), {"name": "OpenHands"})()


def _events():
    return [
        MessageEvent("user", "count the txt files"),
        ActionEvent("terminal", summary="counting txt files"),
        ObservationEvent("terminal", "0"),
        ActionEvent("finish", action=FinishAction("There are 0 txt files.")),
    ]


def test_handler_sets_session_scope_from_conversation_id():
    handler = OpenHandsSpanHandler()
    conversation = Conversation(_events(), conversation_id="conv-abc")
    previous_scope = get_scopes().get(AGENT_SESSION)

    token, _ = handler.pre_tracing({}, None, conversation, (), {})

    assert token is not None
    assert get_scopes().get(AGENT_SESSION) == "conv-abc"

    remove_scope(token)
    assert get_scopes().get(AGENT_SESSION) == previous_scope


def test_handler_ignores_missing_conversation_id():
    handler = OpenHandsSpanHandler()
    conversation = Conversation(_events(), conversation_id=None)

    token, _ = handler.pre_tracing({}, None, conversation, (), {})

    assert token is None


def test_tool_handler_skips_span_without_monocle_context():
    handler = OpenHandsToolHandler()
    assert handler.skip_span({}, None, None, (), {}) is True


def test_tool_handler_keeps_span_with_monocle_context():
    handler = OpenHandsToolHandler()
    span = NonRecordingSpan(
        SpanContext(
            trace_id=0x1, span_id=0x1, is_remote=False, trace_flags=TraceFlags(0x01)
        )
    )
    token = otel_context.attach(set_monocle_span_in_context(span))
    try:
        assert handler.skip_span({}, None, None, (), {}) is False
    finally:
        otel_context.detach(token)


def test_turn_input_and_output():
    arguments = {"instance": Conversation(_events()), "args": (), "kwargs": {}}
    assert _helper.extract_turn_input(arguments) == "count the txt files"
    assert _helper.extract_turn_output(arguments) == "There are 0 txt files."


def test_step_input_and_output():
    conversation = Conversation(_events())
    arguments = {"instance": None, "args": (conversation,), "kwargs": {}}
    # latest observation is what the step responds to; finish message is its outcome
    assert _helper.extract_step_input(arguments) == "0"
    assert _helper.extract_step_output(arguments) == "There are 0 txt files."


def test_tool_accessors():
    action_event = ActionEvent("finish", action=FinishAction("done"))
    arguments = {"instance": None, "args": (action_event,), "kwargs": {}}
    assert _helper.get_tool_name(arguments) == "finish"
    assert _helper.extract_tool_input(arguments) == '{"message": "done"}'

    assert _helper.extract_tool_response([ObservationEvent("terminal", "0")]) == "0"
    assert _helper.extract_tool_response([AgentErrorEvent("boom")]) == "boom"
    assert _helper.extract_tool_response([]) is None


class ParentSpanStub:
    def __init__(self, attributes):
        self.attributes = attributes


def test_source_agent_from_parent_invocation_span():
    parent = ParentSpanStub(
        {"span.type": SPAN_TYPES.AGENTIC_INVOCATION, "entity.1.name": "OpenHands"}
    )
    assert _helper.get_source_agent({"parent_span": parent}) == "OpenHands"
    assert _helper.get_source_agent({"parent_span": None}) is None




import pytest


@pytest.fixture(autouse=True)
def _clear_tool_handler_state():
    OpenHandsToolHandler._thread_hop_contexts.clear()
    OpenHandsToolHandler._async_open_actions.clear()
    yield
    OpenHandsToolHandler._thread_hop_contexts.clear()
    OpenHandsToolHandler._async_open_actions.clear()


def test_tool_handler_restores_stashed_context_across_thread_hop():
    handler = OpenHandsToolHandler()
    action_event = ActionEvent("terminal")
    action_event.id = "evt-1"
    span = NonRecordingSpan(
        SpanContext(trace_id=0x2, span_id=0x2, is_remote=False, trace_flags=TraceFlags(0x01))
    )
    # caller side: stash with a span in context (sync parallel path)
    outer = otel_context.attach(set_monocle_span_in_context(span))
    try:
        to_wrap = {"method": "execute_batch"}
        token, _ = handler.pre_tracing(to_wrap, None, None, ([action_event],), {})
        assert token is None
        assert "evt-1" in OpenHandsToolHandler._thread_hop_contexts
    finally:
        otel_context.detach(outer)

    # thread side: no context — restore from the stash
    to_wrap_run = {"method": "_run_safe"}
    token, _ = handler.pre_tracing(to_wrap_run, None, None, (action_event,), {})
    try:
        assert token is not None
        from monocle_apptrace.instrumentation.common.utils import get_current_monocle_span
        assert get_current_monocle_span().get_span_context().is_valid
        # under an invocation-type parent the span is kept
        assert handler.skip_span(to_wrap_run, None, None, (action_event,), {}) is False
    finally:
        handler.post_tracing(to_wrap_run, None, None, (action_event,), {}, None, token=token)
    assert "evt-1" not in OpenHandsToolHandler._thread_hop_contexts


def test_tool_handler_skips_only_the_matching_async_duplicate():
    handler = OpenHandsToolHandler()
    open_action = ActionEvent("terminal")
    open_action.id = "evt-open"
    other_action = ActionEvent("file_editor")
    other_action.id = "evt-other"

    span = NonRecordingSpan(
        SpanContext(trace_id=0x3, span_id=0x3, is_remote=False, trace_flags=TraceFlags(0x01))
    )
    token = otel_context.attach(set_monocle_span_in_context(span))
    try:
        # _arun_safe opened a span for open_action
        handler.pre_task_processing({"method": "_arun_safe"}, None, None, (open_action,), {}, span)
        assert handler.skip_span({"method": "_run_safe"}, None, None, (open_action,), {}) is True
        # a different action executing on the same thread must NOT be dropped
        assert handler.skip_span({"method": "_run_safe"}, None, None, (other_action,), {}) is False
        # execute_batch never emits a span
        assert handler.skip_span({"method": "execute_batch"}, None, None, (), {}) is True
    finally:
        handler.post_task_processing({"method": "_arun_safe"}, None, None, (open_action,), {}, None, None, span, None)
        otel_context.detach(token)
    assert "evt-open" not in OpenHandsToolHandler._async_open_actions


def test_tool_handler_post_tracing_cleans_leftover_batch_stashes():
    handler = OpenHandsToolHandler()
    cancelled = ActionEvent("terminal")
    cancelled.id = "evt-cancelled"
    handler.pre_tracing({"method": "execute_batch"}, None, None, ([cancelled],), {})
    assert "evt-cancelled" in OpenHandsToolHandler._thread_hop_contexts
    handler.post_tracing({"method": "execute_batch"}, None, None, ([cancelled],), {}, None, token=None)
    assert "evt-cancelled" not in OpenHandsToolHandler._thread_hop_contexts


def test_session_scope_inherited_by_nested_conversation():
    handler = OpenHandsSpanHandler()
    parent = Conversation(_events(), conversation_id="parent-conv")
    child = Conversation(_events(), conversation_id="child-conv")

    token, _ = handler.pre_tracing({}, None, parent, (), {})
    try:
        child_token, _ = handler.pre_tracing({}, None, child, (), {})
        assert child_token is None, "nested conversation must not override the session"
        assert get_scopes().get(AGENT_SESSION) == "parent-conv"
    finally:
        remove_scope(token)
