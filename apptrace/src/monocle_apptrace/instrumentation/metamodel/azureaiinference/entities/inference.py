import logging
from typing import Any

from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamState,
)
from monocle_apptrace.instrumentation.metamodel.azureaiinference import _helper
from monocle_apptrace.instrumentation.common.utils import (
    get_error_message,
    resolve_from_alias,
    get_status,
    get_exception_status_code,
)

logger = logging.getLogger(__name__)


class AzureAIInferenceStreamProcessor(BaseStreamProcessor):
    """Azure AI Inference-specific stream processor."""

    def handle_event(self, item: Any, state: StreamState) -> bool:
        return False

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        if not (hasattr(item, "choices") and item.choices and 
                hasattr(item.choices[0], "delta") and item.choices[0].delta):
            return False

        choice = item.choices[0]
        delta = choice.delta

        if hasattr(delta, "role") and delta.role:
            state.role = delta.role

        if hasattr(delta, "content") and delta.content:
            state.add_content(delta.content)

        return True

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        if not (hasattr(item, "usage") and item.usage):
            return False

        state.token_usage = item.usage
        state.close_stream()
        return True


def process_stream(to_wrap, response, span_processor):
    """Process streaming responses from Azure AI Inference."""
    processor = AzureAIInferenceStreamProcessor()
    processor.process_stream(to_wrap, response, span_processor)


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
