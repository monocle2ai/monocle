from monocle_apptrace.instrumentation.common.constants import SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.fastmcp import _helper

RESOURCES = {
    "type": SPAN_TYPES.AGENTIC_MCP_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_GENERATION,
    "attributes": [
        [
            {
                "_comment": "name of the resource",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_instance_name(arguments),
            },
            {
                "_comment": "resource type",
                "attribute": "type",
                "accessor": lambda arguments: "fastmcp.server",
            },
            {
                "_comment": "resource url",
                "attribute": "url",
                "phase": "post_execution",
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
                    "_comment": "this is Resource URI input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.get_resource_uri(arguments),
                }                
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is Resource content output",
                    "attribute": "output",
                    "accessor": lambda arguments: _helper.get_resource_output(arguments)
                }
            ],
        },
    ],
}


LIST_RESOURCES = {
    "type": SPAN_TYPES.AGENTIC_MCP_INVOCATION,
    "subtype": SPAN_SUBTYPES.COMMUNICATION,
    "attributes": [
        [
            {
                "_comment": "List of resource names",
                "attribute": "resource_name_list",
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
                    "_comment": "list resources input (typically empty)",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.get_list_input(arguments),
                },
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "list of available resources",
                    "attribute": "output",
                    "accessor": lambda arguments: _helper.get_list_output(arguments)
                }
            ],
        },
    ],
}
