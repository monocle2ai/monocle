from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES, SPAN_SUBTYPES
from monocle_apptrace.instrumentation.common.utils import get_error_message
from monocle_apptrace.instrumentation.metamodel.github_copilot import _helper

TOOL = {
    "type": SPAN_TYPES.AGENTIC_TOOL_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_GENERATION,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: _helper.get_tool_type(arguments)},
            {"attribute": "name", "accessor": lambda arguments: _helper.get_tool_name(arguments)},
            {"attribute": "description", "accessor": lambda arguments: _helper.get_tool_description(arguments)},
        ],
        [
            {"attribute": "name", "accessor": lambda arguments: "GitHub Copilot"},
            {"attribute": "type", "accessor": lambda arguments: "agent.github_copilot"},
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_tool_input(arguments),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_tool_response(arguments["result"]),
                },
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
            ],
        },
    ],
}

MCP_TOOL = {
    "type": SPAN_TYPES.AGENTIC_MCP_INVOCATION,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: "tool.mcp"},
            {"attribute": "name", "accessor": lambda arguments: _helper.get_tool_name(arguments)},
            {"attribute": "description", "accessor": lambda arguments: _helper.get_tool_description(arguments)},
        ],
        [
            {"attribute": "name", "accessor": lambda arguments: "GitHub Copilot"},
            {"attribute": "type", "accessor": lambda arguments: "agent.github_copilot"},
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_tool_input(arguments),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_tool_response(arguments["result"]),
                },
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
            ],
        },
    ],
}
