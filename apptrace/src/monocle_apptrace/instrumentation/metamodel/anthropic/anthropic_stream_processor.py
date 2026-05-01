"""Anthropic streaming processor."""

from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamState,
)


class AnthropicStreamProcessor(BaseStreamProcessor):
    """Process Anthropic streaming events into a unified span result."""

    def handle_event(self, item: Any, state: StreamState) -> bool:
        event_type = getattr(item, "type", None)
        if not isinstance(event_type, str):
            return False

        # Handle text deltas from streamed content blocks.
        if event_type == "content_block_delta":
            delta = getattr(item, "delta", None)
            text = getattr(delta, "text", None) if delta is not None else None
            if text:
                state.add_content(text)
            return True

        # Capture tool call metadata when tool_use blocks appear.
        if event_type == "content_block_start":
            block = getattr(item, "content_block", None)
            if getattr(block, "type", None) == "tool_use":
                state.tools.append(
                    {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "arguments": getattr(block, "input", {}),
                    }
                )
            return True

        # Capture initial usage and role metadata.
        if event_type == "message_start":
            message = getattr(item, "message", None)
            usage = getattr(message, "usage", None)
            if usage is not None:
                state.token_usage = {
                    "completion_tokens": getattr(usage, "output_tokens", 0),
                    "prompt_tokens": getattr(usage, "input_tokens", 0),
                    "total_tokens": getattr(usage, "input_tokens", 0)
                    + getattr(usage, "output_tokens", 0),
                }
            role = getattr(message, "role", None)
            if role:
                state.role = role
            return True

        # Capture final stop reason and updated usage.
        if event_type == "message_delta":
            delta = getattr(item, "delta", None)
            usage = getattr(item, "usage", None)
            stop_reason = getattr(delta, "stop_reason", None) if delta is not None else None
            if stop_reason:
                state.finish_reason = stop_reason
            if usage is not None:
                state.token_usage = {
                    "completion_tokens": getattr(usage, "output_tokens", 0),
                    "prompt_tokens": getattr(usage, "input_tokens", 0),
                    "total_tokens": getattr(usage, "input_tokens", 0)
                    + getattr(usage, "output_tokens", 0),
                }
            return True

        # Mark stream completion.
        if event_type == "message_stop":
            state.close_stream()
            return True

        return False

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        return False

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        return False
