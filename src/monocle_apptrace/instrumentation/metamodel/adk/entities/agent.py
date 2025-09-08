from monocle_apptrace.instrumentation.common.constants import SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import get_error_message
from monocle_apptrace.instrumentation.metamodel.adk import _helper
AGENT = {
      "type": SPAN_TYPES.AGENTIC_INVOCATION,
      "subtype": SPAN_SUBTYPES.ROUTING,
      "attributes": [
        [
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.adk'
              },
              {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_agent_name(arguments['instance'])
              },
              {
                "_comment": "agent description",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_agent_description(arguments['instance'])
              },
              {
                "_comment": "delegating agent name",
                "attribute": "from_agent",
                "accessor": lambda arguments: _helper.get_delegating_agent(arguments)
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
                "accessor": lambda arguments: _helper.extract_agent_input(arguments)
            }
          ]
        },
        {
          "name":"data.output",
          "attributes": [
            {
                "_comment": "this is response from LLM",
                "attribute": "response",
                "accessor": lambda arguments: _helper.extract_agent_response(arguments['result'])
            },
            {
                "attribute": "error_code",
                "accessor": lambda arguments: get_error_message(arguments)
            }
          ]
        }
      ]
    }

REQUEST = {
      "type": "agentic.request",
      "attributes": [
        [
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.adk'
              }
        ],
      ],
      "events": [
        {
          "name":"data.input",
          "attributes": [
            {
                "_comment": "this is Agent input",
                "attribute": "input",
                "accessor": lambda arguments: _helper.extract_agent_request_input(arguments)
            }
          ]
        },
        {
          "name":"data.output",
          "attributes": [
            {
                "_comment": "this is response from LLM",
                "attribute": "response",
                "accessor": lambda arguments: _helper.extract_agent_response(arguments['result'])
            }
          ]
        }
      ]
}

DELEGATION = {
      "type": "agentic.delegation",
      "should_skip": lambda arguments: _helper.should_skip_delegation(arguments),
      "attributes": [
        [
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments:'agent.adk'
              },
              {
                "_comment": "name of the agent",
                "attribute": "from_agent",
                "accessor": lambda arguments: _helper.get_delegating_agent(arguments)
              },
              {
                "_comment": "name of the agent called",
                "attribute": "to_agent",
                "accessor": lambda arguments: _helper.get_agent_name(arguments['instance'])
              }
        ]
      ]
}
