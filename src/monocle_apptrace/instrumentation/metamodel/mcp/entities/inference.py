from monocle_apptrace.instrumentation.metamodel.mcp import _helper

TOOLS = {
    "type": "agentic.mcp.invocation",
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
            ],
        },
    ],
}
