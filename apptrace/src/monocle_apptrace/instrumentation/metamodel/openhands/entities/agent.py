from monocle_apptrace.instrumentation.common.constants import SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import get_error_message
from monocle_apptrace.instrumentation.metamodel.openhands import (
    _helper,
)

AGENT_REQUEST = {
      "type": SPAN_TYPES.AGENTIC_REQUEST,
      "subtype": SPAN_SUBTYPES.TURN,
      "attributes": [
        [
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: 'agent.openhands'
              },
              {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_conversation_agent_name(arguments['instance'])
              }
        ],
      ],
      "events": [
        {
          "name": "data.input",
          "attributes": [
            {
                "_comment": "this is the user request for this turn",
                "attribute": "input",
                "accessor": lambda arguments: _helper.extract_turn_input(arguments)
            }
          ]
        },
        {
          "name": "data.output",
          "attributes": [
            {
                "attribute": "error_code",
                "accessor": lambda arguments: get_error_message(arguments)
            },
            {
                "_comment": "this is the final agent response for this turn",
                "attribute": "response",
                "accessor": lambda arguments: _helper.extract_turn_output(arguments)
            }
          ]
        }
      ]
}

AGENT = {
      "type": SPAN_TYPES.AGENTIC_INVOCATION,
      "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
      "attributes": [
        [
              {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: 'agent.openhands'
              },
              {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_agent_name(arguments['instance'])
              }
        ]
      ],
      "events": [
        {
          "name": "data.input",
          "attributes": [
            {
                "_comment": "this is Agent input",
                "attribute": "input",
                "accessor": lambda arguments: _helper.extract_step_input(arguments)
            }
          ]
        },
        {
          "name": "data.output",
          "attributes": [
            {
                "attribute": "error_code",
                "accessor": lambda arguments: get_error_message(arguments)
            },
            {
                "_comment": "this is what the agent produced in this step",
                "attribute": "response",
                "accessor": lambda arguments: _helper.extract_step_output(arguments)
            }
          ]
        }
      ]
}
