import logging
import time
from types import SimpleNamespace
from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.azureaiinference import _helper
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message,
    resolve_from_alias, 
    patch_instance_method,
    get_status,
    get_exception_status_code
)

logger = logging.getLogger(__name__)


def process_stream(to_wrap, response, span_processor):
    """Process streaming responses from Azure AI Inference."""
    waiting_for_first_token = True
    stream_start_time = time.time_ns()
    first_token_time = stream_start_time
    stream_closed_time = None
    accumulated_response = ""
    token_usage = None
    role = "assistant"

    # For sync iteration - patch __next__ instead of __iter__
    if to_wrap and hasattr(response, "__next__"):
        original_next = response.__next__

        def new_next(self):
            nonlocal waiting_for_first_token, first_token_time, stream_closed_time, accumulated_response, token_usage, role

            try:
                item = original_next()
                
                # Handle Azure AI Inference streaming chunks
                if hasattr(item, 'choices') and item.choices:
                    choice = item.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'role') and choice.delta.role:
                        role = choice.delta.role
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                        if waiting_for_first_token:
                            waiting_for_first_token = False
                            first_token_time = time.time_ns()

                        accumulated_response += choice.delta.content
                
                # Check for usage information at the end of stream
                if hasattr(item, 'usage') and item.usage:
                    token_usage = item.usage
                    stream_closed_time = time.time_ns()

                return item

            except StopIteration:
                # Stream is complete, process final span
                if span_processor:
                    ret_val = SimpleNamespace(
                        type="stream",
                        role=role,
                        timestamps={
                            "data.input": int(stream_start_time),
                            "data.output": int(first_token_time),
                            "metadata": int(stream_closed_time or time.time_ns()),
                        },
                        output_text=accumulated_response,
                        usage=token_usage,
                    )
                    span_processor(ret_val)
                raise
            except Exception as e:
                logger.warning(
                    "Warning: Error occurred while processing item in new_next: %s",
                    str(e),
                )
                raise

        patch_instance_method(response, "__next__", new_next)
        
    # For async iteration - patch __anext__ instead of __aiter__
    if to_wrap and hasattr(response, "__anext__"):
        original_anext = response.__anext__

        async def new_anext(self):
            nonlocal waiting_for_first_token, first_token_time, stream_closed_time, accumulated_response, token_usage, role

            try:
                item = await original_anext()
                
                # Handle Azure AI Inference streaming chunks
                if hasattr(item, 'choices') and item.choices:
                    choice = item.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'role') and choice.delta.role:
                        role = choice.delta.role
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                        if waiting_for_first_token:
                            waiting_for_first_token = False
                            first_token_time = time.time_ns()

                        accumulated_response += choice.delta.content
                
                # Check for usage information at the end of stream
                if hasattr(item, 'usage') and item.usage:
                    token_usage = item.usage
                    stream_closed_time = time.time_ns()

                return item

            except StopAsyncIteration:
                # Stream is complete, process final span
                if span_processor:
                    ret_val = SimpleNamespace(
                        type="stream",
                        role=role,
                        timestamps={
                            "data.input": int(stream_start_time),
                            "data.output": int(first_token_time),
                            "metadata": int(stream_closed_time or time.time_ns()),
                        },
                        output_text=accumulated_response,
                        usage=token_usage,
                    )
                    span_processor(ret_val)
                raise
            except Exception as e:
                logger.warning(
                    "Warning: Error occurred while processing item in new_anext: %s",
                    str(e),
                )
                raise

        patch_instance_method(response, "__anext__", new_anext)


INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "is_auto_close": lambda kwargs: kwargs.get("stream", False) is False,
    "response_processor": process_stream,
    "attributes": [
        [
            {
                "_comment": "Azure AI Inference provider type, endpoint",
                "attribute": "type",
                "accessor": lambda arguments: f"inference.{_helper.get_inference_type(arguments)}"
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: _helper.get_provider_name(arguments['instance'])
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: _helper.extract_inference_endpoint(arguments['instance'])
            },
            {
                "attribute": "deployment",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments['instance'].__dict__,
                    ['deployment', 'deployment_name', 'azure_deployment', '_deployment']
                )
            }
        ],
        [
            {
                "_comment": "LLM Model information",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_model_name(arguments)
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: f"model.llm.{_helper.get_model_name(arguments)}" if _helper.get_model_name(arguments) else "model.llm.unknown"
            }
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
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "Chat messages input to Azure AI Inference",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_messages(arguments['kwargs'])
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "Response from Azure AI Inference",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(arguments)
                },
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments)
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "Usage metadata from Azure AI Inference",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments['result'], 
                        arguments.get('instance')
                    )
                },
                {
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(arguments)
                },
                {
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(
                        _helper.extract_finish_reason(arguments)
                    )
                }
            ]
        }
    ]
}
