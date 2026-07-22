from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamEventTypes,
    StreamState,
)

class OpenAIStreamProcessor(BaseStreamProcessor):
    """OpenAI-specific stream processor."""
    
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
            if hasattr(item, "response") and hasattr(item.response, "output_text"):
                if not state.accumulated_response and item.response.output_text:
                    state.accumulated_response = item.response.output_text
            if hasattr(item, "response") and hasattr(item.response, "output") and isinstance(item.response.output, list):
                has_function_call = False
                for output_item in item.response.output:
                    if getattr(output_item, "type", "") == "function_call":
                        has_function_call = True
                        state.tools.append(
                            {
                                "id": getattr(output_item, "id", ""),
                                "name": getattr(output_item, "name", ""),
                                "arguments": getattr(output_item, "arguments", ""),
                            }
                        )
                if has_function_call:
                    state.finish_reason = "tool_calls"

            if hasattr(item, "response") and hasattr(item.response, "status") and not state.finish_reason:
                response_status = getattr(item.response, "status", None)
                if response_status == "completed":
                    state.finish_reason = "stop"
                elif response_status == "incomplete":
                    incomplete_details = getattr(item.response, "incomplete_details", None)
                    state.finish_reason = getattr(incomplete_details, "reason", None) or "length"
        
        return True

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        """Handle chunked streaming with delta objects (choices[0].delta format)."""
        if not (hasattr(item, "choices") and item.choices and
                hasattr(item.choices[0], "delta") and item.choices[0].delta):
            return False

        choice = item.choices[0]
        delta = choice.delta

        # Handle role
        if hasattr(delta, "role") and delta.role:
            state.role = delta.role

        # Handle content
        if hasattr(delta, "content") and delta.content:
            state.add_content(delta.content)

        # Handle refusal (new field)
        if hasattr(delta, "refusal") and delta.refusal:
            state.refusal = delta.refusal

        return True

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        """Handle final chunk with usage info and completion metadata."""
        if not (hasattr(item, "object") and item.object == StreamEventTypes.CHAT_COMPLETION_CHUNK and
                hasattr(item, "usage") and item.usage):
            return False

        state.token_usage = item.usage
        state.close_stream()

        # Capture finish_reason from the chunk
        if (hasattr(item, "choices") and item.choices and len(item.choices) > 0 and
            hasattr(item.choices[0], "finish_reason") and item.choices[0].finish_reason):
            state.finish_reason = item.choices[0].finish_reason

        return True

    def assemble_data(self, state: StreamState) -> None:
        """Assemble tool calls and completion data from fragmented streaming chunks."""
        for item in state.raw_items:
            try:
                if (hasattr(item, "choices") and item.choices and
                    isinstance(item.choices, list) and len(item.choices) > 0):

                    choice = item.choices[0]

                    # Extract tool calls
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