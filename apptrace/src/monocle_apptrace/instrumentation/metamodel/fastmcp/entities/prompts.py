from monocle_apptrace.instrumentation.common.constants import SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.fastmcp import _helper

PROMPTS = {
    "type": SPAN_TYPES.AGENTIC_MCP_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_GENERATION,
    "attributes": [
        [
            {
                "_comment": "name of the prompt",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_instance_name(arguments),
            },
            {
                "_comment": "prompt type",
                "attribute": "type",
                "accessor": lambda arguments: "fastmcp.server",
            },
            {
                "_comment": "prompt url",
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
                    "_comment": "this is Prompt input arguments",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.get_params_arguments(arguments),
                },
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is Prompt messages output",
                    "attribute": "output",
                    "accessor": lambda arguments: _helper.get_prompt_output(arguments)
                }
            ],
        },
    ],
}


LIST_PROMPTS = {
    "type": SPAN_TYPES.AGENTIC_MCP_INVOCATION,
    "subtype": SPAN_SUBTYPES.COMMUNICATION,
    "attributes": [
        [
            {
                "_comment": "List of prompt names",
                "attribute": "prompt_name_list",
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
                    "_comment": "list prompts input (typically empty)",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.get_list_input(arguments),
                },
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "list of available prompts",
                    "attribute": "output",
                    "accessor": lambda arguments: _helper.get_list_output(arguments)
                },
            ],
        },
    ],
}
