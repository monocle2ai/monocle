"""Unit tests for Anthropic cached token extraction (non-streaming and streaming)."""
import unittest
from types import SimpleNamespace

from monocle_apptrace.instrumentation.metamodel.anthropic._helper import (
    update_span_from_llm_response,
)
from monocle_apptrace.instrumentation.metamodel.anthropic.anthropic_stream_processor import (
    AnthropicStreamProcessor,
)
from monocle_apptrace.instrumentation.common.stream_processor import StreamState


def _make_response(input_tokens=100, output_tokens=50, total_tokens=150,
                   cache_read_input_tokens=None):
    """Simulate an Anthropic Messages response object (attribute-style usage)."""
    usage_kwargs = dict(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    if cache_read_input_tokens is not None:
        usage_kwargs["cache_read_input_tokens"] = cache_read_input_tokens
    return SimpleNamespace(usage=SimpleNamespace(**usage_kwargs))


class TestBasicTokenExtraction(unittest.TestCase):
    def test_basic_tokens(self):
        result = update_span_from_llm_response(_make_response(80, 40, 120))
        self.assertEqual(result["prompt_tokens"], 80)
        self.assertEqual(result["completion_tokens"], 40)
        self.assertEqual(result["total_tokens"], 120)
        self.assertNotIn("cache_read_input_tokens", result)

    def test_no_response(self):
        self.assertEqual(update_span_from_llm_response(None), {})


class TestCachedTokens(unittest.TestCase):
    def test_cache_read_captured(self):
        result = update_span_from_llm_response(
            _make_response(200, 60, 260, cache_read_input_tokens=80))
        self.assertEqual(result["cache_read_input_tokens"], 80)
        self.assertEqual(result["prompt_tokens"], 200)

    def test_zero_cache_read_still_captured(self):
        """0 is a real value (no cache hit this call) and should be captured."""
        result = update_span_from_llm_response(
            _make_response(cache_read_input_tokens=0))
        self.assertEqual(result.get("cache_read_input_tokens"), 0)

    def test_no_cache_read_field(self):
        result = update_span_from_llm_response(_make_response())
        self.assertNotIn("cache_read_input_tokens", result)

    def test_dict_usage_passthrough(self):
        """A dict-form usage is spread through verbatim (langchain/serialized path)."""
        response = SimpleNamespace(usage={"prompt_tokens": 100, "completion_tokens": 50,
                                           "cache_read_input_tokens": 30})
        result = update_span_from_llm_response(response)
        self.assertEqual(result["cache_read_input_tokens"], 30)


class TestStreamingCachedTokens(unittest.TestCase):
    """Cache-read is reported at message_start and must survive message_delta."""

    def _event(self, event_type, **kwargs):
        return SimpleNamespace(type=event_type, **kwargs)

    def test_cache_read_survives_message_delta(self):
        processor = AnthropicStreamProcessor()
        state = StreamState()

        # message_start carries the cache-read count.
        processor.handle_event(
            self._event(
                "message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=200, output_tokens=0,
                                          cache_read_input_tokens=150),
                    role="assistant",
                ),
            ),
            state,
        )
        self.assertEqual(state.token_usage["cache_read_input_tokens"], 150)

        # message_delta overwrites usage (no cache field) — count must be preserved.
        processor.handle_event(
            self._event(
                "message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(input_tokens=0, output_tokens=42),
            ),
            state,
        )
        self.assertEqual(state.token_usage["cache_read_input_tokens"], 150)
        self.assertEqual(state.token_usage["completion_tokens"], 42)

    def test_no_cache_read_when_absent(self):
        processor = AnthropicStreamProcessor()
        state = StreamState()
        processor.handle_event(
            self._event(
                "message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=100, output_tokens=0),
                    role="assistant",
                ),
            ),
            state,
        )
        self.assertNotIn("cache_read_input_tokens", state.token_usage)


if __name__ == "__main__":
    unittest.main()
