from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES, SPAN_SUBTYPES
from monocle_apptrace.instrumentation.common.utils import get_span_id, get_error_message
from monocle_apptrace.instrumentation.metamodel.claude_cli import _helper


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
            {"attribute": "type", "accessor": lambda arguments: "inference.anthropic"},
            {"attribute": "provider_name", "accessor": lambda arguments: "api.anthropic.com"},
            {"attribute": "inference_endpoint", "accessor": lambda arguments: "https://api.anthropic.com"},
        ],
        [
            {"attribute": "name", "accessor": lambda arguments: arguments["kwargs"].get("model", "claude")},
            {"attribute": "type", "accessor": lambda arguments: "model.llm." + arguments["kwargs"].get("model", "claude")},
        ],
        # Entity 3: the tool dispatched in this round (empty string = no entity created)
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
            {"attribute": "type", "accessor": lambda arguments: "agent.claude_cli"},
            {"attribute": "name", "accessor": lambda arguments: "Claude Code"},
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
                },
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
            ],
        },
    ],
}

INVOCATION = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: "agent.claude_cli"},
            {"attribute": "name", "accessor": lambda arguments: "Claude Code"},
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
                },
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
            ],
        },
    ],
}

SUBAGENT_INVOCATION = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: "agent.claude_cli"},
            {
                "attribute": "name",
                "accessor": lambda arguments: arguments["kwargs"].get("agent_type", "agent"),
            },
            {
                "attribute": "description",
                "accessor": lambda arguments: arguments["kwargs"].get("description", ""),
            },
            {"attribute": "from_agent", "accessor": lambda arguments: "Claude Code"},
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
