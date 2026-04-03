"""Mistral-specific streaming processor for chat.stream and chat.stream_async."""

from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamState,
)


class MistralStreamProcessor(BaseStreamProcessor):
    """Stream processor for Mistral CompletionEvent chunks."""

    def handle_event(self, item: Any, state: StreamState) -> bool:
        # Mistral streaming does not use response.* event envelopes.
        return False

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        payload = getattr(item, "data", item)

        if not (hasattr(payload, "choices") and payload.choices):
            return False

        choice = payload.choices[0]
        delta = getattr(choice, "delta", None)
        found = False

        if delta is not None:
            role = getattr(delta, "role", None)
            if role:
                state.role = role

            content = getattr(delta, "content", None)
            if content:
                state.add_content(content)
                found = True

            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    fn = getattr(tool_call, "function", None)
                    if fn is None:
                        continue
                    state.tools.append(
                        {
                            "id": getattr(tool_call, "id", None),
                            "name": getattr(fn, "name", ""),
                            "arguments": getattr(fn, "arguments", ""),
                        }
                    )
                found = True

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            state.finish_reason = finish_reason
            # A finish_reason on chunk indicates stream completion semantics.
            state.close_stream()
            found = True

        return found

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        payload = getattr(item, "data", item)
        usage = getattr(payload, "usage", None) or getattr(item, "usage", None)
        if usage is None:
            return False

        state.token_usage = usage
        state.close_stream()
        return True
