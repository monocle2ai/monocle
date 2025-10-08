from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import get_error_message, resolve_from_alias
from monocle_apptrace.instrumentation.metamodel.hugging_face import _helper


INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "subtype": lambda arguments: _helper.agent_inference_type(arguments),
    "attributes": [
        [
            {
                "_comment": "provider type, name, deployment, inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: "inference.huggingface"
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: "huggingface"
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: "https://api-inference.huggingface.co/v1/"
            }
        ],
        [
            {
                "_comment": "LLM Model (repo ID on Hugging Face hub)",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["kwargs"],
                    ["model", "model_name", "endpoint_name", "deployment_name"]
                )
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: (
                    "model.llm." +
                    resolve_from_alias(
                        arguments["kwargs"],
                        ["model", "model_name", "endpoint_name", "deployment_name"]
                    )
                )
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is instruction and user query to LLM",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_messages(arguments["kwargs"])
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments)
                },
                {
                    "_comment": "result from Hugging Face inference",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(arguments)
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM, includes token counts",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments.get("result"),
                        include_token_counts=True  # new flag for streaming handling
                    )
                },
                {
                    "_comment": "finish reason from Hugging Face response",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(arguments)
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(_helper.extract_finish_reason(arguments))
                }
            ]
        }
    ]
}