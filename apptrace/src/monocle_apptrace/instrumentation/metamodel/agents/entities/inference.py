from monocle_apptrace.instrumentation.common.constants import AGENT_REQUEST_SPAN_NAME, SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import get_error_message
from monocle_apptrace.instrumentation.metamodel.agents import _helper
from monocle_apptrace.instrumentation.common.utils import get_error_message

AGENT = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.ROUTING,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: _helper.AGENTS_AGENT_NAME_KEY,
            },
            {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_agent_name(arguments),
            },
            {
                "_comment": "agent description",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_agent_description(arguments),
            },
            {
                "_comment": "agent instructions",
                "attribute": "instructions",
                "accessor": lambda arguments: _helper.get_agent_instructions(arguments),
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
                    "accessor": lambda arguments: _helper.extract_agent_input(
                        arguments
                    ),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments)
                },
                {
                    "_comment": "this is response from Agent",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(
                        arguments["result"]
                    ),
            },
            {
                "attribute": "error_code",
                "accessor": lambda arguments: get_error_message(arguments)
                }
            ],
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata from Agent response",
                    "accessor": lambda arguments: _helper.update_span_from_agent_response(
                        arguments["result"]
                    ),
                }
            ],
        },
    ],
}

AGENT_REQUEST = {
    "type": SPAN_TYPES.AGENTIC_REQUEST,
    "subtype": SPAN_SUBTYPES.PLANNING,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: _helper.AGENTS_AGENT_NAME_KEY,
            }
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is Agent input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_agent_input(
                        arguments
                    ),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is response from Agent",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(
                        arguments["result"]
                    ),
            },
            {
                "attribute": "error_code",
                "accessor": lambda arguments: get_error_message(arguments)
            }],
        },
    ],
}

TOOLS = {
    "type": SPAN_TYPES.AGENTIC_TOOL_INVOCATION,
    "subtype": SPAN_SUBTYPES.ROUTING,
    "attributes": [
        [
            {
                "_comment": "tool type",
                "attribute": "type",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.get_tool_type(arguments["span"]),
            },
            {
                "_comment": "name of the tool",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_tool_name(
                    arguments["instance"]
                ),
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
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_source_agent(),
            },
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: _helper.AGENTS_AGENT_NAME_KEY,
            },
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is Tool input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_tool_input(arguments),
                },
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is response from Tool",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_tool_response(
                        arguments["result"]
                    ),
                },
                {
                "attribute": "error_code",
                "accessor": lambda arguments: get_error_message(arguments)
                }
            ],
        },
    ],
}

AGENT_DELEGATION = {
    "type": SPAN_TYPES.AGENTIC_DELEGATION,
    "subtype": SPAN_SUBTYPES.ROUTING,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: _helper.AGENTS_AGENT_NAME_KEY,
            },
            {
                "_comment": "name of the source agent",
                "attribute": "from_agent",
                "accessor": lambda arguments: _helper.get_source_agent(),
            },
            {
                "_comment": "name of the target agent",
                "attribute": "to_agent",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.extract_handoff_target(arguments),
            },
        ]
    ],
}
