from monocle_apptrace.instrumentation.metamodel.langgraph import (
    _helper
)
AGENT = {
      "type": "agent",
      "attributes": [
        [
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.langgraph'
              },
              {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments:arguments['instance'].name
              }
        ]
      ],
      "events": [
        {
          "name":"data.input",
          "attributes": [
            {
                "_comment": "this is Agent input",
                "attribute": "query",
                "accessor": lambda arguments: _helper.extract_input(arguments)
            }
          ]
        },
        {
          "name":"data.output",
          "attributes": [
            {
                "_comment": "this is response from LLM",
                "attribute": "response",
                "accessor": lambda arguments: _helper.handle_openai_response(arguments['result'])
            }
          ]
        }
      ]
    }

TOOLS = {
      "type": "agent.tool",
      "attributes": [
        [
              {
                "_comment": "tool type",
                "attribute": "type",
                "accessor": lambda arguments:'tool.langgraph'
              },
              {
                "_comment": "name of the tool",
                "attribute": "name",
                "accessor": lambda arguments:arguments['instance'].name
              },
              {
                  "_comment": "tool description",
                  "attribute": "description",
                  "accessor": lambda arguments: arguments['instance'].description
              }
        ]
      ],
      "events": [
        {
          "name":"data.input",
          "attributes": [
            {
                "_comment": "this is Tool input",
                "attribute": "Inputs",
                "accessor": lambda arguments: _helper.get_tool_args(arguments) 
            }
          ]
        },
        {
          "name":"data.output",
          "attributes": [
            {
                "_comment": "this is response from Tool",
                "attribute": "response",
                "accessor": lambda arguments: _helper.extract_response(arguments['result'])
            },
            {
                "_comment": "this is status from Tool",
                "attribute": "status",
                "accessor": lambda arguments: _helper.get_status(arguments['result'])
            }
          ]
        }
      ]
}
