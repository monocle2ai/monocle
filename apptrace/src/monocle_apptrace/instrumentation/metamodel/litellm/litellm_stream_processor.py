"""LiteLLM streaming processor for OpenAI and Azure OpenAI streaming responses."""

from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamState,
)


class LiteLLMStreamProcessor(BaseStreamProcessor):
    """Stream processor for LiteLLM streaming responses.

    LiteLLM streaming yields response chunks where:
    - Each chunk has partial text in choices[0].delta.content
    - The final chunk carries finish_reason in choices[0].finish_reason
    - Token usage is typically not available in stream chunks
    - Both text and finish_reason can appear in the same (final) chunk
    """

    def handle_event(self, item: Any, state: StreamState) -> bool:
        # LiteLLM does not use SSE-style response.* events
        return False

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        """Extract partial text from a LiteLLM streaming chunk."""
        found = False

        # Preferred path: choices[0].delta.content
        if hasattr(item, "choices") and item.choices:
            choice = item.choices[0]
            if hasattr(choice, "delta"):
                delta = choice.delta
                if hasattr(delta, "content") and delta.content:
                    state.add_content(delta.content)
                    found = True
                # Check for tool calls in delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if hasattr(tool_call, "function"):
                            func = tool_call.function
                            state.tools.append({
                                "name": getattr(func, "name", ""),
                                "args": getattr(func, "arguments", {}),
                            })
                            state.finish_reason = "tool_calls"
                            found = True

        return found

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        """Capture finish_reason and token usage from final LiteLLM chunks."""
        found = False

        # Usage may be present on a terminal stream chunk when include_usage=True.
        usage = getattr(item, "usage", None)
        if usage is not None:
            def _to_int(value: Any) -> int:
                try:
                    return int(value) if value is not None else 0
                except (TypeError, ValueError):
                    return 0

            def _pick(source: Any, keys: list[str]) -> int:
                for key in keys:
                    if isinstance(source, dict) and key in source and source.get(key) is not None:
                        return _to_int(source.get(key))
                    if hasattr(source, key):
                        value = getattr(source, key)
                        if value is not None:
                            return _to_int(value)
                return 0

            completion_tokens = _pick(usage, ["completion_tokens", "output_tokens", "candidates_token_count"])
            prompt_tokens = _pick(usage, ["prompt_tokens", "input_tokens", "prompt_token_count"])
            total_tokens = _pick(usage, ["total_tokens", "total_token_count"])
            if total_tokens == 0 and (completion_tokens or prompt_tokens):
                total_tokens = completion_tokens + prompt_tokens

            state.token_usage = {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            }
            found = True

        # Finish reason usually comes on choices[0].finish_reason in a terminal chunk.
        if hasattr(item, "choices") and item.choices:
            choice = item.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None:
                state.finish_reason = finish_reason
                state.close_stream()
                found = True

        return found

    def try_framework_specific_processing(self, item: Any, state: StreamState) -> bool:
        """Override to allow both text extraction AND finish_reason capture on the same chunk."""
        has_chunk = self.handle_chunk(item, state)
        has_completion = self.handle_completion(item, state)
        return has_chunk or has_completion
