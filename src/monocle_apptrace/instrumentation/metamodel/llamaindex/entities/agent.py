from monocle_apptrace.instrumentation.common.constants import AGENT_REQUEST_SPAN_NAME, SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.llamaindex import (
    _helper,
)

AGENT = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.ROUTING,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.llamaindex'
            },
            {
                "_comment": "Agent name, type and Tools.",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_agent_name(arguments['instance'])
            },
            {
                "_comment": "agent description",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_agent_description(arguments['instance'])
            }
        ]

    ],
    "events": [
        {"name": "data.input",
         "attributes": [

             {
                 "_comment": "this is instruction and user query to LLM",
                 "attribute": "input",
                 "accessor": lambda arguments: _helper.extract_agent_input(arguments['args'])
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
    "type": AGENT_REQUEST_SPAN_NAME,
    "subtype": SPAN_SUBTYPES.PLANNING,
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
      "type": SPAN_TYPES.AGENTIC_TOOL_INVOCATION,
      "subtype": SPAN_SUBTYPES.ROUTING,
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
                "accessor": lambda arguments: _helper.get_tool_name(arguments['args'], arguments['instance'])
              },
              {
                  "_comment": "tool description",
                  "attribute": "description",
                  "accessor": lambda arguments: _helper.get_tool_description(arguments)
              }
        ],
        [             
              {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_source_agent(arguments['parent_span'])
              },          
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.llamaindex'
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
                "accessor": lambda arguments: _helper.extract_tool_args(arguments) 
            }
          ]
        },
        {
          "name":"data.output",
          "attributes": [
            {
                "_comment": "this is response from Tool",
                "attribute": "response",
                "accessor": lambda arguments: _helper.extract_tool_response(arguments['result'])
             }
          ]
        }
      ]
}

AGENT_DELEGATION = {
    "type": SPAN_TYPES.AGENTIC_DELEGATION,
    "subtype": SPAN_SUBTYPES.ROUTING,
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
                "accessor": lambda arguments: _helper.get_source_agent(arguments['parent_span'])
              },
              {
                "_comment": "name of the agent called",
                "attribute": "to_agent",
                "accessor": lambda arguments: _helper.get_target_agent(arguments['result'])
              }
        ]
      ]
}