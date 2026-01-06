"""Inference entity definitions for Microsoft Agent Framework."""

from monocle_apptrace.instrumentation.common.constants import (
    AGENT_REQUEST_SPAN_NAME,
    SPAN_SUBTYPES,
    SPAN_TYPES,
)
from monocle_apptrace.instrumentation.metamodel.msagent import _helper
from monocle_apptrace.instrumentation.common.utils import get_error_message

# Turn-level request span (agentic.request with turn subtype)
# For Microsoft Agent Framework, turn doesn't include agent name (follows ADK pattern)
AGENT_REQUEST = {
    "type": SPAN_TYPES.AGENTIC_REQUEST,
    "subtype": SPAN_SUBTYPES.TURN,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.microsoft"
            },
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is Agent input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_request_agent_input(arguments)
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is response from Agent",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(
                        arguments["result"], arguments.get("span"), arguments.get("instance"), arguments.get("kwargs")
                    )
                }
            ]
        }
    ]
}

# Agent invocation span (agentic.invocation with content_processing subtype)
# Used for chat client get_response/get_streaming_response
AGENT = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.microsoft",
            },
            {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_chat_client_name(arguments["instance"]),
            },
            {
                "_comment": "model id",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_chat_client_model(arguments["instance"]),
            },
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is Agent input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_chat_client_input(arguments),
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
                    "_comment": "this is response from Agent",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_chat_client_response(
                        arguments["result"], arguments.get("span")
                    ),
                },
            ],
        },
    ],
}

# Agent orchestrator span (agentic.invocation with routing subtype)
# Used when agent makes routing decisions (tool calls, etc.)
AGENT_ORCHESTRATOR = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.ROUTING,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.microsoft",
            },
            {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_agent_name(arguments["instance"]),
            },
            {
                "_comment": "agent instructions/system prompt",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_agent_instructions(
                    arguments["instance"]
                ),
            },
        ]
    ],
    "events": []
}

TOOL = {
    "type": SPAN_TYPES.AGENTIC_TOOL_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_GENERATION,
    "attributes": [
        [
            {
                "_comment": "tool type",
                "attribute": "type",
                "accessor": lambda arguments: "tool.microsoft",
            },
            {
                "_comment": "name of the tool",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_tool_name(arguments["instance"]),
            },
            {
                "_comment": "tool description",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_tool_description(
                    arguments["instance"]
                ),
            },
        ],
        [
            {
                "_comment": "agent name (owner of the tool)",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_agent_name_from_context(),
            },
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.microsoft",
            },
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is Tool input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_tool_input(arguments),
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
                    "_comment": "this is response from Tool",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_tool_response(
                        arguments["result"], arguments.get("span")
                    ),
                },
            ],
        },
    ],
}
