from monocle_apptrace.instrumentation.common.constants import SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import get_error_message, get_span_id
from monocle_apptrace.instrumentation.metamodel.adk import _helper
from monocle_apptrace.instrumentation.common.utils import get_error_message
AGENT = {
      "type": SPAN_TYPES.AGENTIC_INVOCATION,
      "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
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
              },
              {
                "_comment": "last agent invocation id",
                "attribute": "from_agent_span_id",
                "accessor": lambda arguments: get_span_id(arguments['parent_span'])
              }
        ]
      ],
      "events": [
        {
          "name":"data.input",
          "attributes": [
            {
                "_comment": "this is Agent input",
                "attribute": "input",
                "accessor": lambda arguments: _helper.extract_agent_input(arguments)
            }
          ]
        },
        {
          "name":"data.output",
          "attributes": [
            {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments)
            },
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

AGENT_ORCHESTRATOR = {
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
      ]
    }

REQUEST = {
      "type": SPAN_TYPES.AGENTIC_REQUEST,
      "subtype": SPAN_SUBTYPES.TURN,
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
