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
        """Extract partial text from a LiteLLM streaming chunk.

        Tool calls are NOT assembled here — they arrive as fragments across
        multiple chunks and are reconstructed in assemble_data() after the
        stream completes.
        """
        if not (hasattr(item, "choices") and item.choices):
            return False

        choice = item.choices[0]
        if not (hasattr(choice, "delta") and hasattr(choice.delta, "content") and choice.delta.content):
            return False

        state.add_content(choice.delta.content)
        return True

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

    def assemble_data(self, state: StreamState) -> None:
        """Assemble tool call name/args from fragmented streaming chunks.

        LiteLLM mirrors OpenAI's streaming format: function name arrives in the
        first tool_call delta; arguments accumulate across subsequent deltas.
        We iterate raw_items here to reconstruct each tool call correctly.
        """
        tool_call_map: dict = {}  # index -> {name, args_parts}
        for item in state.raw_items:
            try:
                if not (hasattr(item, "choices") and item.choices):
                    continue
                choice = item.choices[0]
                if not (hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls):
                    continue
                for tool_call in choice.delta.tool_calls:
                    idx = getattr(tool_call, "index", 0)
                    if idx not in tool_call_map:
                        tool_call_map[idx] = {"name": "", "args_parts": []}
                    if hasattr(tool_call, "function") and tool_call.function:
                        func = tool_call.function
                        if getattr(func, "name", None):
                            tool_call_map[idx]["name"] = func.name
                        if getattr(func, "arguments", None):
                            tool_call_map[idx]["args_parts"].append(func.arguments)
                # Capture finish_reason
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason:
                    state.finish_reason = finish_reason
            except Exception as e:
                self.logger.warning("Warning: Error assembling tool call data: %s", str(e))

        if tool_call_map:
            # Replace any partially assembled tools with fully assembled ones
            state.tools = [
                {"name": tc["name"], "args": "".join(tc["args_parts"])}
                for tc in tool_call_map.values()
            ]

    def try_framework_specific_processing(self, item: Any, state: StreamState) -> bool:
        """Override to allow both text extraction AND finish_reason capture on the same chunk."""
        has_chunk = self.handle_chunk(item, state)
        has_completion = self.handle_completion(item, state)
        return has_chunk or has_completion
