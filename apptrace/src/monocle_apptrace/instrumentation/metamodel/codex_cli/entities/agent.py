from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES, SPAN_SUBTYPES
from monocle_apptrace.instrumentation.common.utils import get_span_id
from monocle_apptrace.instrumentation.metamodel.codex_cli import _helper


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
            {"attribute": "type", "accessor": lambda arguments: "inference.openai"},
            {"attribute": "provider_name", "accessor": lambda arguments: "api.openai.com"},
            {"attribute": "inference_endpoint", "accessor": lambda arguments: "https://api.openai.com"},
        ],
        [
            {"attribute": "name", "accessor": lambda arguments: arguments["kwargs"].get("model", "codex")},
            {"attribute": "type", "accessor": lambda arguments: "model.llm." + arguments["kwargs"].get("model", "codex")},
        ],
        [
            {"attribute": "name", "accessor": lambda arguments: arguments["kwargs"].get("tool_name", "")},
            {"attribute": "type", "accessor": lambda arguments: _tool_entity_type(arguments["kwargs"].get("tool_name", ""))},
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
            {"attribute": "type", "accessor": lambda arguments: "agent.codex_cli"},
            {"attribute": "name", "accessor": lambda arguments: "Codex CLI"},
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
            {"attribute": "type", "accessor": lambda arguments: "agent.codex_cli"},
            {"attribute": "name", "accessor": lambda arguments: "Codex CLI"},
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

SUBAGENT_INVOCATION = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: "agent.codex_cli"},
            {
                "attribute": "name",
                "accessor": lambda arguments: arguments["kwargs"].get("agent_nickname") or arguments["kwargs"].get("agent_role", "agent"),
            },
            {
                "attribute": "role",
                "accessor": lambda arguments: arguments["kwargs"].get("agent_role", ""),
            },
            {"attribute": "from_agent", "accessor": lambda arguments: "Codex CLI"},
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
