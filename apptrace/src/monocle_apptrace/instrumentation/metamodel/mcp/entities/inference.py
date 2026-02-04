from monocle_apptrace.instrumentation.common.constants import SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import get_error_message
from monocle_apptrace.instrumentation.metamodel.mcp import _helper

TOOLS = {
    "type": SPAN_TYPES.AGENTIC_MCP_INVOCATION,
    "subtype": SPAN_SUBTYPES.ROUTING,
    "attributes": [
        [
            {
                "_comment": "name of the tool",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_name(arguments),
            },
            {
                "_comment": "tool type",
                "attribute": "type",
                "accessor": lambda arguments: "mcp.server",
            },
            {
                "_comment": "tool url",
                "attribute": "url",
                "accessor": lambda arguments: _helper.get_url(arguments),
            },
            {
                "_comment": "server name",
                "attribute": "server_name",
                "accessor": lambda arguments: _helper.get_server_name(arguments),
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
                    "accessor": lambda arguments: _helper.get_output_text(arguments)
                },
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments)
                }
            ],
        },
    ],
}
