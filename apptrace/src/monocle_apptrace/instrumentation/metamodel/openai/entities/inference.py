import logging
from typing import Any

from monocle_apptrace.instrumentation.common.constants import (
    SPAN_TYPES,
)
from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamState,
)
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message,
    resolve_from_alias,
)
from monocle_apptrace.instrumentation.metamodel.openai import (
    _helper,
)

logger = logging.getLogger(__name__)

# Streaming Event Type Constants
class StreamEventTypes:
    """Constants for streaming event types."""
    # Event-based streaming events
    RESPONSE_OUTPUT_TEXT_DELTA = "response.output_text.delta"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_FUNCTION_CALL_DELTA = "response.function_call.delta"
    
    # Object types
    CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
    
    # Response prefixes
    RESPONSE_PREFIX = "response."
    
    @classmethod
    def is_response_event(cls, event_type: str) -> bool:
        """Check if an event type is a response event."""
        return isinstance(event_type, str) and event_type.startswith(cls.RESPONSE_PREFIX)


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
                if (hasattr(item, 'choices') and item.choices and 
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


def process_stream(to_wrap, response, span_processor):
    """Process OpenAI streaming responses using the generic processor."""
    processor = OpenAIStreamProcessor()
    processor.process_stream(to_wrap, response, span_processor)


INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "subtype": lambda arguments: _helper.agent_inference_type(arguments),
    "is_auto_close": lambda kwargs: kwargs.get("stream", False) is False,
    "response_processor": process_stream,
    "attributes": [
        [
            {
                "_comment": "provider type ,name , deployment , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: "inference."
                + (_helper.get_inference_type(arguments["instance"]))
                or "openai",
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: _helper.extract_provider_name(
                    arguments["instance"]
                ),
            },
            {
                "attribute": "deployment",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["instance"].__dict__,
                    [
                        "engine",
                        "azure_deployment",
                        "deployment_name",
                        "deployment_id",
                        "deployment",
                    ],
                ),
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["instance"].__dict__,
                    ["azure_endpoint", "api_base", "endpoint"],
                )
                or _helper.extract_inference_endpoint(arguments["instance"]),
            },
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["kwargs"],
                    ["model", "model_name", "endpoint_name", "deployment_name"],
                ),
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: "model.llm."
                + resolve_from_alias(
                    arguments["kwargs"],
                    ["model", "model_name", "endpoint_name", "deployment_name"],
                ),
            },
        ],
        [
            {
                "_comment": "Tool name when finish_type is tool_call",
                "attribute": "name",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.extract_tool_name(arguments),
            },
            {
                "_comment": "Tool type when finish_type is tool_call", 
                "attribute": "type",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.extract_tool_type(arguments),
            },
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is instruction and user query to LLM",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_messages(
                        arguments["kwargs"]
                    ),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
                {
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(
                        arguments,
                    ),
                },
            ],
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments["result"]
                    ),
                },
                {
                    "_comment": "finish reason from OpenAI response",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(
                        arguments
                    ),
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(
                        _helper.extract_finish_reason(arguments)
                    ),
                }
            ],
        },
    ],
}
