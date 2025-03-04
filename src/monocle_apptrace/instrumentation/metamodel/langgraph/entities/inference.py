from monocle_apptrace.instrumentation.metamodel.langgraph import (
    _helper
)
INFERENCE = {
      "type": "agent",
      "attributes": [
        [
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.oai'
              },
              {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments:arguments['instance'].name
              },
              {
                "_comment": "agent tools",
                "attribute": "tools",
                "accessor": lambda arguments: _helper.tools(arguments['instance'])
              }
        ]
      ],
      "events": [
        {
          "name":"data.input",
          "attributes": [
            {
                "_comment": "this is LLM input",
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
       },
       {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(arguments['result'])
                }
            ]
        }
      ]
    }