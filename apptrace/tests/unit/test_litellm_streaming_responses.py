from types import SimpleNamespace

from monocle_apptrace.instrumentation.common.stream_processor import StreamState
from monocle_apptrace.instrumentation.metamodel.litellm import _helper
from monocle_apptrace.instrumentation.metamodel.litellm.entities.inference import _is_streaming
from monocle_apptrace.instrumentation.metamodel.litellm.entities.responses import (
    _is_streaming as _responses_is_streaming,
)
from monocle_apptrace.instrumentation.metamodel.openai.openai_stream_processor import (
    OpenAIStreamProcessor,
)


def _chunk(delta=None, usage=None, finish_reason=None):
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        object="chat.completion.chunk",
        choices=[choice] if delta is not None or finish_reason else [],
        usage=usage,
    )


def _tool_delta(index, id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    tool_call = SimpleNamespace(index=index, id=id, function=function)
    return SimpleNamespace(content=None, role=None, refusal=None, tool_calls=[tool_call])


def test_stream_processor_accumulates_tool_arguments_across_fragments():
    processor = OpenAIStreamProcessor()
    state = StreamState()
    for delta in [
        _tool_delta(0, id="call_1", name="terminal", arguments=""),
        _tool_delta(0, arguments='{"command":'),
        _tool_delta(0, arguments='"date"}'),
    ]:
        processor.process_fragment(_chunk(delta=delta), state)
    processor.assemble_data(state)
    assert state.tools == [
        {"id": "call_1", "name": "terminal", "arguments": '{"command":"date"}'}
    ]


def test_stream_processor_captures_usage_on_delta_carrying_chunk():
    # litellm's CustomStreamWrapper attaches usage to a chunk that still has an
    # (empty) delta, so handle_completion never fires — handle_chunk must catch it.
    processor = OpenAIStreamProcessor()
    state = StreamState()
    usage = SimpleNamespace(completion_tokens=9, prompt_tokens=9, total_tokens=18)
    empty_delta = SimpleNamespace(content=None, role=None, refusal=None, tool_calls=None)
    processor.process_fragment(_chunk(delta=empty_delta, usage=usage), state)
    assert state.token_usage is usage


def test_extract_assistant_message_stream_result():
    result = SimpleNamespace(
        type="stream",
        tools=[{"id": "call_1", "name": "finish", "arguments": '{"message":"done"}'}],
        output_text="",
    )
    arguments = {"result": result, "exception": None}
    message = _helper.extract_assistant_message(arguments)
    assert '"tool_name": "finish"' in message
    assert '"tool_arguments": "{\\"message\\":\\"done\\"}"' in message


def test_is_streaming_reads_optional_params():
    assert _is_streaming({"optional_params": {"stream": True}}) is True
    assert _is_streaming({"optional_params": {}}) is False
    assert _is_streaming({}) is False


def test_responses_is_streaming_reads_request_params():
    assert _responses_is_streaming({"response_api_optional_request_params": {"stream": True}}) is True
    assert _responses_is_streaming({}) is False


def _responses_result():
    function_call = SimpleNamespace(
        type="function_call", call_id="call_9", name="terminal", arguments='{"command":"ls"}'
    )
    message = SimpleNamespace(
        type="message", content=[SimpleNamespace(text="listing files")]
    )
    return SimpleNamespace(
        output=[function_call, message],
        status="completed",
        usage=SimpleNamespace(completion_tokens=None, output_tokens=5, prompt_tokens=None,
                              input_tokens=7, total_tokens=12),
    )


def test_extract_responses_input_and_output():
    kwargs = {"input": [{"role": "user", "content": "list the files"}]}
    assert _helper.extract_responses_input(kwargs) == ['{"user": "list the files"}']

    arguments = {"result": _responses_result(), "exception": None}
    output = _helper.extract_responses_output(arguments)
    assert '"tool_name": "terminal"' in output

    assert _helper.extract_responses_finish_reason(arguments) == "tool_calls"

    meta = _helper.update_span_from_llm_response(arguments["result"])
    assert meta["completion_tokens"] == 5
    assert meta["prompt_tokens"] == 7
    assert meta["total_tokens"] == 12


def test_extract_responses_finish_reason_without_tools():
    message_only = SimpleNamespace(
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(text="hi")])],
        status="completed",
    )
    arguments = {"result": message_only, "exception": None}
    assert _helper.extract_responses_finish_reason(arguments) == "completed"


def test_inference_streaming_variant_never_auto_closes():
    from monocle_apptrace.instrumentation.metamodel.litellm.entities.inference import (
        INFERENCE,
        INFERENCE_STREAMING,
    )
    # async_streaming has no stream flag in optional_params; the variant must
    # defer close unconditionally while sharing the base entity's processors.
    assert INFERENCE_STREAMING["is_auto_close"]({}) is False
    assert INFERENCE_STREAMING["attributes"] is INFERENCE["attributes"]
    assert INFERENCE_STREAMING["events"] is INFERENCE["events"]


class _SelfIterStream:
    """Mimics litellm's CustomStreamWrapper: __iter__ returns self, items via __next__."""

    def __init__(self, items):
        self._items = iter(items)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._items)


def _text_chunk(text):
    delta = SimpleNamespace(content=text, role="assistant", refusal=None, tool_calls=None)
    return _chunk(delta=delta)


def test_process_stream_direct_next_consumption_closes_once():
    # litellm pulls via next(self.completion_stream) without touching __iter__
    processor = OpenAIStreamProcessor()
    fired = []
    stream = _SelfIterStream([_text_chunk("hel"), _text_chunk("lo")])
    wrapper = processor.process_stream({"x": 1}, stream, fired.append)
    assert wrapper is None  # patched in place
    collected = ""
    try:
        while True:
            collected += stream.__next__().choices[0].delta.content
    except StopIteration:
        pass
    # post-exhaustion re-iteration must not re-fire the close
    for _ in range(3):
        try:
            stream.__next__()
        except StopIteration:
            pass
    assert collected == "hello"
    assert len(fired) == 1
    assert fired[0].output_text == "hello"


def test_process_stream_for_loop_consumption_closes_once():
    processor = OpenAIStreamProcessor()
    fired = []
    stream = _SelfIterStream([_text_chunk("a"), _text_chunk("b")])
    processor.process_stream({"x": 1}, stream, fired.append)
    text = "".join(ch.choices[0].delta.content for ch in stream)
    assert text == "ab"
    assert len(fired) == 1
    assert fired[0].output_text == "ab"


def test_assemble_data_indexless_argument_fragments():
    # Some providers emit argument fragments with index=None; they belong to
    # the most recently started tool call, not a new ghost entry.
    processor = OpenAIStreamProcessor()
    state = StreamState()
    frags = [
        _tool_delta(0, id="call_1", name="terminal", arguments=""),
        _tool_delta(None, arguments='{"cmd":'),
        _tool_delta(None, arguments='"ls"}'),
    ]
    for delta in frags:
        processor.process_fragment(_chunk(delta=delta), state)
    processor.assemble_data(state)
    assert state.tools == [{"id": "call_1", "name": "terminal", "arguments": '{"cmd":"ls"}'}]


def test_responses_sync_handler_skips_async_shim():
    from monocle_apptrace.instrumentation.metamodel.litellm.litellm_span_handler import (
        LiteLLMSyncSpanHandler,
    )
    handler = LiteLLMSyncSpanHandler()
    # response_api_handler with _is_async=True returns an unawaited coroutine;
    # the sync span must be skipped (the async wrap owns the real span)
    assert handler.skip_span({}, None, None, (), {"_is_async": True}) is True
    assert handler.skip_span({}, None, None, (), {"_is_async": False}) is False
    assert handler.skip_span({}, None, None, (), {"acompletion": True}) is True


def test_handle_event_assembles_streamed_responses_tool_calls():
    processor = OpenAIStreamProcessor()
    state = StreamState()
    added = SimpleNamespace(
        type="response.output_item.added",
        item=SimpleNamespace(type="function_call", call_id="call_9", name="terminal", arguments=""),
    )
    delta1 = SimpleNamespace(type="response.function_call_arguments.delta", delta='{"cmd":')
    delta2 = SimpleNamespace(type="response.function_call_arguments.delta", delta='"date"}')
    for ev in (added, delta1, delta2):
        processor.process_fragment(ev, state)
    assert state.tools == [{"id": "call_9", "name": "terminal", "arguments": '{"cmd":"date"}'}]
    # finish reason for a stream namespace with tools
    ns = SimpleNamespace(type="stream", tools=state.tools)
    assert _helper.extract_responses_finish_reason({"result": ns, "exception": None}) == "tool_calls"


def test_extract_assistant_message_non_streaming_tool_calls():
    # tool-call responses have content=None; the output must carry the tool
    # calls instead of {"assistant": null}
    function = SimpleNamespace(name="terminal", arguments='{"command":"ls"}')
    tool_call = SimpleNamespace(id="call_7", function=function)
    message = SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    out = _helper.extract_assistant_message({"result": response, "exception": None})
    assert '"tool_name": "terminal"' in out
    assert "null" not in out

    # plain content responses are unchanged
    message2 = SimpleNamespace(role="assistant", content="hello", tool_calls=None)
    response2 = SimpleNamespace(choices=[SimpleNamespace(message=message2)])
    out2 = _helper.extract_assistant_message({"result": response2, "exception": None})
    assert out2 == '{"assistant": "hello"}'
