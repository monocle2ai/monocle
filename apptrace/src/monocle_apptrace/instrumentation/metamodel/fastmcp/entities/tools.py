from monocle_apptrace.instrumentation.common.constants import SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.fastmcp import _helper

TOOLS = {
    "type": SPAN_TYPES.AGENTIC_MCP_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_GENERATION,
    "attributes": [
        [
            {
                "_comment": "name of the tool",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_instance_name(arguments),
            },
            {
                "_comment": "tool type",
                "attribute": "type",
                "accessor": lambda arguments: "fastmcp.server",
            },
            {
                "_comment": "tool url",
                "attribute": "url",
                "accessor": lambda arguments: _helper.get_url(arguments),
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
                    "accessor": lambda arguments: _helper.get_params_arguments(
                        arguments
                    ),
                },
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is Tool output",
                    "attribute": "output",
                    "accessor": lambda arguments: _helper.get_tool_output(arguments)
                }
            ],
        },
    ],
}

LIST_TOOLS = {
    "type": SPAN_TYPES.AGENTIC_MCP_INVOCATION,
    "subtype": SPAN_SUBTYPES.COMMUNICATION,
    "attributes": [
        [
            {
                "_comment": "List of tool names",
                "attribute": "tool_name_list",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.get_list_of_names(arguments),
            },
            {
                "_comment": "operation type",
                "attribute": "type",
                "accessor": lambda arguments: "fastmcp.server",
            },
            {
                "_comment": "server url",
                "attribute": "url",
                "accessor": lambda arguments: _helper.get_url(arguments),
            },
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "list tools input (typically empty)",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.get_list_input(arguments),
                },
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "list of available tools",
                    "attribute": "output",
                    "accessor": lambda arguments: _helper.get_list_output(arguments)
                }
            ],
        },
    ],
}