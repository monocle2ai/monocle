from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamEventTypes,
    StreamState,
)


def _is_function_call(content: Any) -> bool:
    return getattr(content, "type", None) == "function_call"


def _is_usage(content: Any) -> bool:
    return getattr(content, "type", None) == "usage"


class MSAgentStreamProcessor(BaseStreamProcessor):
    """Microsoft Agent Framework-specific stream processor."""

    def handle_event(self, item: Any, state: StreamState) -> bool:
        """Handle Server-Sent Events with response.* event types."""
        if not (hasattr(item, "type") and StreamEventTypes.is_response_event(item.type)):
            return False

        state.update_first_token_time()

        if item.type == StreamEventTypes.RESPONSE_OUTPUT_TEXT_DELTA:
            state.accumulated_response += item.delta
        elif item.type == StreamEventTypes.RESPONSE_TEXT_DELTA:
            state.accumulated_response += item.delta
        elif item.type == StreamEventTypes.RESPONSE_COMPLETED:
            state.close_stream()
            if hasattr(item, "response") and hasattr(item.response, "usage"):
                state.token_usage = item.response.usage

        return True

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        """Handle MS Agent streaming chunks (ChatResponseUpdate / AgentRunResponseUpdate)
        and OpenAI-style chunks with delta objects."""
        # MS Agent ChatResponseUpdate / AgentRunResponseUpdate with .contents
        if hasattr(item, "contents") and isinstance(item.contents, list):
            handled = False
            for content in item.contents:
                if _is_function_call(content):
                    state.tools.append({
                        "id": getattr(content, "call_id", ""),
                        "name": getattr(content, "name", ""),
                        "arguments": getattr(content, "arguments", ""),
                    })
                    handled = True
                elif _is_usage(content) and hasattr(content, "usage_details"):
                    state.token_usage = content.usage_details
                    handled = True

            # Extract text (property that concatenates TextContent items)
            if hasattr(item, "text") and item.text:
                state.add_content(item.text)
                handled = True

            # Extract finish_reason from ChatResponseUpdate
            if hasattr(item, "finish_reason") and item.finish_reason:
                fr = item.finish_reason
                state.finish_reason = getattr(fr, "value", str(fr))
                handled = True

            if handled:
                return True

        # Simple .text attribute (AgentRunResponseUpdate without .contents)
        if hasattr(item, "text") and item.text and not hasattr(item, "contents"):
            state.add_content(item.text)
            return True

        # OpenAI-style chunks with choices[0].delta (from _inner_get_streaming_response)
        if (hasattr(item, "choices") and item.choices and
                hasattr(item.choices[0], "delta") and item.choices[0].delta):
            choice = item.choices[0]
            delta = choice.delta

            if hasattr(delta, "role") and delta.role:
                state.role = delta.role

            if hasattr(delta, "content") and delta.content:
                state.add_content(delta.content)

            return True

        return False

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        """Handle final chunk with usage info and completion metadata."""
        # OpenAI-style chat.completion.chunk
        if (hasattr(item, "object") and item.object == StreamEventTypes.CHAT_COMPLETION_CHUNK and
                hasattr(item, "usage") and item.usage):
            state.token_usage = item.usage
            state.close_stream()

            if (hasattr(item, "choices") and item.choices and len(item.choices) > 0 and
                    hasattr(item.choices[0], "finish_reason") and item.choices[0].finish_reason):
                state.finish_reason = item.choices[0].finish_reason
            return True

        return False

    def assemble_data(self, state: StreamState) -> None:
        """Assemble tool calls and completion data from stored raw items."""
        for item in state.raw_items:
            try:
                if (hasattr(item, "choices") and item.choices and
                        isinstance(item.choices, list) and len(item.choices) > 0):

                    choice = item.choices[0]

                    # Extract tool calls from OpenAI-style chunks
                    if (hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls") and
                            choice.delta.tool_calls):
                        for tool_call in choice.delta.tool_calls:
                            if (hasattr(tool_call, "id") and tool_call.id and
                                    hasattr(tool_call, "function") and tool_call.function):
                                state.tools.append({
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "arguments": getattr(tool_call.function, "arguments", ""),
                                })

                    # Extract finish_reason
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        state.finish_reason = choice.finish_reason

            except Exception as e:
                self.logger.warning(
                    "Warning: Error occurred while processing tool calls: %s", str(e)
                )

        # Default finish_reason when the stream completed with text but no
        # explicit finish_reason was received (common with Assistants API
        # streaming where finish_reason is not set on individual updates).
        if not state.finish_reason:
            if state.tools:
                state.finish_reason = "tool_calls"
            elif state.accumulated_response:
                state.finish_reason = "stop"
