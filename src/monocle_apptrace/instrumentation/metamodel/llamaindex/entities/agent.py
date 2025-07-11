from monocle_apptrace.instrumentation.metamodel.llamaindex import (
    _helper,
)

AGENT = {
    "type": "agentic.invocation",
    "attributes": [
        [
            {
                "_comment": "Agent name, type and Tools.",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_agent_name(arguments['instance'])
            },
            {
                "_comment": "agent description",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_agent_description(arguments['instance'])
            },
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.llamaindex'
            }
        ]

    ],
    "events": [
        {"name": "data.input",
         "attributes": [

             {
                 "_comment": "this is instruction and user query to LLM",
                 "attribute": "input",
                 "accessor": lambda arguments: _helper.extract_agent_args(arguments['args'])
             }
         ]
         },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is response from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(arguments)
                }
            ]
        }
    ]
} 

AGENT_REQUEST = {
    "type": "agentic.request",
    "attributes": [
        [
              {
                "_comment": "agent request type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.llamaindex'
              }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is request to LLM",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_agent_request_input(arguments['kwargs'])
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is response from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_request_output(arguments)
                }
            ]
        }
    ]
}

TOOLS = {
      "type": "agentic.tool",
      "attributes": [
        [
              {
                "_comment": "tool type",
                "attribute": "type",
                "accessor": lambda arguments:'tool.llamaindex'
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
                "accessor": lambda arguments: _helper.get_tool_response(arguments['result'])
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

AGENT_DELEGATION = {
    "type": "agentic.delegation",
      "attributes": [
        [
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.llamaindex'
              },
              {
                "_comment": "name of the agent",
                "attribute": "from_agent",
                "accessor": lambda arguments: _helper.get_source_agent_name(arguments['parent_span'])
              },
              {
                "_comment": "name of the agent called",
                "attribute": "to_agent",
                "accessor": lambda arguments: _helper.get_delegated_agent_name(arguments['kwargs'])
              }
        ]
      ]
}