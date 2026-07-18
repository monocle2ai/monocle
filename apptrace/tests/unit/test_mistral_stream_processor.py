"""
Offline unit tests for Mistral streaming instrumentation.

These drive fake Mistral CompletionEvent chunks through the real
MistralStreamProcessor and the _helper extractors, so the streaming trace path
can be validated without a live MISTRAL_API_KEY / network call.
"""
from types import SimpleNamespace

import pytest

from monocle_apptrace.instrumentation.metamodel.mistral.mistral_stream_processor import (
    MistralStreamProcessor,
)
from monocle_apptrace.instrumentation.metamodel.mistral import _helper


def _chunk(content=None, role=None, finish_reason=None, usage=None, tool_calls=None):
    """Build a fake mistralai CompletionEvent: chunk.data.choices[0].delta.*"""
    delta = SimpleNamespace(content=content, role=role, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    data = SimpleNamespace(choices=[choice], usage=usage)
    return SimpleNamespace(data=data)


def _run_stream(chunks):
    """Feed chunks through the processor and return the built span result."""
    processor = MistralStreamProcessor()
    state = processor.initialize_state(0)
    for chunk in chunks:
        processor.process_fragment(chunk, state)
    return processor.create_span_result(state, 0)


class TestMistralStreamProcessor:
    def test_accumulates_text_across_chunks(self):
        result = _run_stream([
            _chunk(content="The moon ", role="assistant"),
            _chunk(content="is about "),
            _chunk(content="384,400 km away."),
            _chunk(finish_reason="stop",
                   usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20)),
        ])
        assert result.output_text == "The moon is about 384,400 km away."
        assert result.finish_reason == "stop"
        assert result.timestamps["role"] == "assistant"

    def test_captures_tool_calls(self):
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}'),
        )
        result = _run_stream([
            _chunk(role="assistant", tool_calls=[tool_call]),
            _chunk(finish_reason="tool_calls"),
        ])
        assert result.tools is not None and len(result.tools) == 1
        assert result.tools[0]["name"] == "get_weather"
        assert result.tools[0]["arguments"] == '{"city":"Paris"}'
        assert result.finish_reason == "tool_calls"

    def test_captures_usage_from_final_chunk(self):
        result = _run_stream([
            _chunk(content="hi", role="assistant"),
            _chunk(finish_reason="stop",
                   usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7)),
        ])
        assert result.usage is not None
        assert result.usage.total_tokens == 7


class TestExtractorsAgainstStreamResult:
    """The entity accessors run against the SimpleNamespace produced above."""

    def _text_result(self):
        return _run_stream([
            _chunk(content="Hello ", role="assistant"),
            _chunk(content="world"),
            _chunk(finish_reason="stop",
                   usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)),
        ])

    def test_extract_assistant_message_returns_text(self):
        result = self._text_result()
        assert _helper.extract_assistant_message({"result": result}) == "Hello world"

    def test_extract_output_text(self):
        result = self._text_result()
        assert _helper.extract_output_text({"result": result}) == "Hello world"

    def test_extract_finish_reason_from_stream_result(self):
        result = self._text_result()
        assert _helper.extract_finish_reason({"result": result}) == "stop"
        assert _helper.map_finish_reason_to_finish_type("stop") == "success"

    def test_update_span_tokens_from_stream_result(self):
        result = self._text_result()
        tokens = _helper.update_span_from_llm_response(result, include_token_counts=True)
        assert tokens == {"completion_tokens": 2, "prompt_tokens": 3, "total_tokens": 5}

    def test_tool_call_message_and_finish_type(self):
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}'),
        )
        result = _run_stream([
            _chunk(role="assistant", tool_calls=[tool_call]),
            _chunk(finish_reason="tool_calls"),
        ])
        # finish_type derived from the stream finish_reason
        assert _helper.map_finish_reason_to_finish_type(
            _helper.extract_finish_reason({"result": result})
        ) == "tool_call"
        # tool name resolvable from the streamed tool dict
        assert _helper.extract_tool_name({"result": result}) == "get_weather"

    def test_stream_tool_call_subtype(self):
        """agent_inference_type must classify a streamed tool_calls result as a tool call."""
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}'),
        )
        result = _run_stream([
            _chunk(role="assistant", tool_calls=[tool_call]),
            _chunk(finish_reason="tool_calls"),
        ])
        from monocle_apptrace.instrumentation.common.constants import INFERENCE_TOOL_CALL
        assert _helper.agent_inference_type({"result": result}) == INFERENCE_TOOL_CALL

    def test_stream_text_subtype_is_turn_end(self):
        from monocle_apptrace.instrumentation.common.constants import INFERENCE_TURN_END
        assert _helper.agent_inference_type({"result": self._text_result()}) == INFERENCE_TURN_END


class TestNonStreamingToolName:
    """Regression guard: extract_tool_name must handle both a streamed tool dict
    and a non-streaming ToolCall *object* (the getter loop must skip a None result
    from the dict-getter and fall through to tc.function.name)."""

    def _obj_result(self):
        # Mimic mistralai ChatCompletionResponse with a ToolCall object.
        tool_call = SimpleNamespace(function=SimpleNamespace(name="get_current_weather"))
        message = SimpleNamespace(tool_calls=[tool_call], content="")
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        return SimpleNamespace(choices=[choice])

    def test_object_tool_call_name(self):
        assert _helper.extract_tool_name({"result": self._obj_result()}) == "get_current_weather"

    def test_dict_tool_call_name(self):
        result = _run_stream([
            _chunk(role="assistant", tool_calls=[
                SimpleNamespace(id="c1", function=SimpleNamespace(name="get_weather", arguments="{}"))
            ]),
            _chunk(finish_reason="tool_calls"),
        ])
        assert _helper.extract_tool_name({"result": result}) == "get_weather"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
