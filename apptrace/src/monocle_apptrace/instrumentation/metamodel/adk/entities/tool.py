from monocle_apptrace.instrumentation.common.constants import SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import get_error_message
from monocle_apptrace.instrumentation.metamodel.adk import _helper
TOOL = {
      "type": SPAN_TYPES.AGENTIC_TOOL_INVOCATION,
      "subtype": SPAN_SUBTYPES.CONTENT_GENERATION,
      "attributes": [
        [
              {
                "_comment": "tool type",
                "attribute": "type",
                "accessor": lambda arguments:'tool.adk'
              },
              {
                "_comment": "name of the tool",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_tool_name(arguments['instance'])
              },
              {
                  "_comment": "tool description",
                  "attribute": "description",
                  "accessor": lambda arguments: _helper.get_tool_description(arguments['instance'])
              }
        ],
        [
              {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_source_agent(arguments)
              },
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.adk'
              }
        ]
      ],
      "events": [
        {
          "name":"data.input",
          "attributes": [
            {
                "_comment": "this is Tool input",
                "attribute": "input",
                "accessor": lambda arguments: _helper.extract_tool_input(arguments) 
            },
          ]
        },
        {
          "name":"data.output",
          "attributes": [
            {
                "_comment": "this is response from Tool",
                "attribute": "response",
                "accessor": lambda arguments: _helper.extract_tool_response(arguments['result'])
            },
            {
                "attribute": "error_code",
                "accessor": lambda arguments: get_error_message(arguments)
            },
          ]
        }
      ]
}
