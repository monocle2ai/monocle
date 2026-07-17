"""Unit tests for the non-streaming litellm additions in this PR:

- ``extract_assistant_message`` surfaces tool calls (was ``{"assistant": null}``)
- Responses API accessors (input / output / finish_reason / inference type)
- ``litellm_sync_handler`` skips the async Responses shim (``_is_async``)

All stand-ins — no network, no litellm/openhands install required.
"""
from types import SimpleNamespace

from monocle_apptrace.instrumentation.metamodel.litellm import _helper
from monocle_apptrace.instrumentation.metamodel.litellm.litellm_span_handler import (
    LiteLLMSyncSpanHandler,
)
from monocle_apptrace.instrumentation.metamodel.litellm.entities.responses import RESPONSES


# --- non-streaming tool-call output fix -------------------------------------

def test_extract_assistant_message_non_streaming_tool_calls():
    # tool-call responses carry content=None; the output must surface the tool
    # calls instead of {"assistant": null}
    function = SimpleNamespace(name="terminal", arguments='{"command":"ls"}')
    tool_call = SimpleNamespace(id="call_7", function=function)
    message = SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    out = _helper.extract_assistant_message({"result": response, "exception": None})
    assert '"tool_name": "terminal"' in out
    assert "null" not in out


def test_extract_assistant_message_plain_content_unchanged():
    message = SimpleNamespace(role="assistant", content="hello", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    out = _helper.extract_assistant_message({"result": response, "exception": None})
    assert out == '{"assistant": "hello"}'


# --- Responses API accessors ------------------------------------------------

def _responses_result():
    function_call = SimpleNamespace(
        type="function_call", call_id="call_9", name="terminal", arguments='{"command":"ls"}'
    )
    message = SimpleNamespace(type="message", content=[SimpleNamespace(text="listing files")])
    return SimpleNamespace(
        output=[function_call, message],
        status="completed",
        usage=SimpleNamespace(completion_tokens=None, output_tokens=5, prompt_tokens=None,
                              input_tokens=7, total_tokens=12),
    )


def test_extract_responses_input_string_and_messages():
    assert _helper.extract_responses_input({"input": "hi there"}) == ['{"user": "hi there"}']
    assert _helper.extract_responses_input(
        {"input": [{"role": "user", "content": "list the files"}]}
    ) == ['{"user": "list the files"}']


def test_extract_responses_output_and_finish_reason_tool_calls():
    arguments = {"result": _responses_result(), "exception": None}
    out = _helper.extract_responses_output(arguments)
    assert '"tool_name": "terminal"' in out
    assert _helper.extract_responses_finish_reason(arguments) == "tool_calls"
    assert _helper.responses_inference_type(arguments) == _helper.INFERENCE_TOOL_CALL


def test_extract_responses_finish_reason_message_only():
    message_only = SimpleNamespace(
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(text="hi")])],
        status="completed",
    )
    arguments = {"result": message_only, "exception": None}
    assert _helper.extract_responses_finish_reason(arguments) == "completed"
    assert _helper.responses_inference_type(arguments) == _helper.INFERENCE_TURN_END


def test_responses_usage_metadata():
    meta = _helper.update_span_from_llm_response(_responses_result())
    assert meta["completion_tokens"] == 5
    assert meta["prompt_tokens"] == 7
    assert meta["total_tokens"] == 12


# --- async Responses shim skip ----------------------------------------------

def test_responses_sync_handler_skips_async_shim():
    # response_api_handler with _is_async=True returns an unawaited coroutine;
    # the sync span must be skipped (the async wrap owns the real span)
    handler = LiteLLMSyncSpanHandler()
    assert handler.skip_span({}, None, None, (), {"_is_async": True}) is True
    assert handler.skip_span({}, None, None, (), {"_is_async": False}) is False
    assert handler.skip_span({}, None, None, (), {"acompletion": True}) is True


# --- the Responses entity is non-streaming ----------------------------------

def test_responses_entity_is_non_streaming():
    # no deferred-close / stream-processor wiring on this entity
    assert "response_processor" not in RESPONSES
    assert "is_auto_close" not in RESPONSES
