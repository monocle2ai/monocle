from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES, SPAN_SUBTYPES
from monocle_apptrace.instrumentation.common.utils import get_span_id
from monocle_apptrace.instrumentation.metamodel.copilot_cli import _helper


def _get_inference_type(model: str) -> str:
    return "inference.anthropic" if model.startswith("claude") else "inference.openai"


def _get_provider_name(model: str) -> str:
    return "api.anthropic.com" if model.startswith("claude") else "api.openai.com"


def _get_inference_endpoint(model: str) -> str:
    return "https://api.anthropic.com" if model.startswith("claude") else "https://api.openai.com"


def _tool_entity_type(tool_name: str) -> str:
    if not tool_name:
        return ""
    if tool_name.startswith("mcp__"):
        return "tool.mcp"
    return "tool.function"


INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: _get_inference_type(arguments["kwargs"].get("model", ""))},
            {"attribute": "provider_name", "accessor": lambda arguments: _get_provider_name(arguments["kwargs"].get("model", ""))},
            {"attribute": "inference_endpoint", "accessor": lambda arguments: _get_inference_endpoint(arguments["kwargs"].get("model", ""))},
        ],
        [
            {"attribute": "name", "accessor": lambda arguments: arguments["kwargs"].get("model", "copilot")},
            {"attribute": "type", "accessor": lambda arguments: "model.llm." + arguments["kwargs"].get("model", "copilot")},
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: arguments["kwargs"].get("input_text", ""),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: arguments["kwargs"].get("output_text", ""),
                }
            ],
        },
        {
            "name": "metadata",
            "attributes": [
                {"accessor": lambda arguments: arguments["kwargs"].get("tokens", {})},
                {
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: arguments["kwargs"].get("finish_reason", ""),
                },
                {
                    "attribute": "finish_type",
                    "accessor": lambda arguments: arguments["kwargs"].get("finish_type", ""),
                },
            ],
        },
    ],
}

REQUEST = {
    "type": SPAN_TYPES.AGENTIC_REQUEST,
    "subtype": SPAN_SUBTYPES.TURN,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: "agent.copilot_cli"},
            {"attribute": "name", "accessor": lambda arguments: "Copilot CLI"},
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_agent_request_input(arguments),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(arguments["result"]),
                }
            ],
        },
    ],
}

INVOCATION = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: "agent.copilot_cli"},
            {"attribute": "name", "accessor": lambda arguments: "Copilot CLI"},
            {"attribute": "from_agent_span_id", "accessor": lambda arguments: get_span_id(arguments["parent_span"])},
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: arguments["kwargs"].get("prompt", ""),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(arguments["result"]),
                }
            ],
        },
    ],
}

SESSION_SUMMARY = {
    "type": SPAN_TYPES.AGENTIC_REQUEST,
    "subtype": SPAN_SUBTYPES.COMMUNICATION,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: "agent.copilot_cli.session"},
            {"attribute": "name", "accessor": lambda arguments: "Copilot CLI Session"},
        ],
        [
            {"attribute": "name", "accessor": lambda arguments: arguments["kwargs"].get("model", "copilot")},
            {"attribute": "type", "accessor": lambda arguments: "model.llm." + arguments["kwargs"].get("model", "copilot")},
        ],
    ],
    "events": [
        {
            "name": "metadata",
            "attributes": [
                {"accessor": lambda arguments: arguments["kwargs"].get("totals") or {}},
            ],
        },
    ],
}
