import logging

from monocle_apptrace.instrumentation.common.constants import AGENT_REQUEST_SPAN_NAME, SPAN_SUBTYPES, SPAN_TYPES
from monocle_apptrace.instrumentation.metamodel.llamaindex import (
    _helper,
)

logger = logging.getLogger(__name__)


def process_workflow_handler(to_wrap, response, span_processor):
    """Deferred span completion for ``Workflow.run``.

    ``run`` returns a ``WorkflowHandler`` (a Future) synchronously while the work
    runs in a background task, so keep the span open (``is_auto_close`` False)
    and close it via a done-callback. The handler is returned unchanged so both
    ``await wf.run(...)`` and the streaming-handler pattern keep working.
    """
    if response is None or not hasattr(response, "add_done_callback"):
        span_processor(response)
        return response

    def _on_workflow_done(future):
        result = None
        try:
            result = future.result()
        except Exception:
            result = None
        try:
            span_processor(result)
        except Exception as e:
            logger.warning("Warning: Error closing workflow span: %s", str(e))

    response.add_done_callback(_on_workflow_done)
    return response

AGENT = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
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
            },
            {
                "_comment": "delegating agent name",
                "attribute": "from_agent",
                "accessor": lambda arguments: _helper.get_from_agent_name()
            },
            {
                "_comment": "from_agent invocation id",
                "attribute": "from_agent_span_id",
                "accessor": lambda arguments: _helper.get_from_agent_span_id()
            }
        ]

    ],
    "events": [
        {"name": "data.input",
         "attributes": [

             {
                 "_comment": "this is instruction and user query to LLM",
                 "attribute": "input",
                 "accessor": lambda arguments: _helper.extract_agent_input(arguments['args'], arguments['kwargs'])
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
    "type": SPAN_TYPES.AGENTIC_REQUEST,
    "subtype": SPAN_SUBTYPES.TURN,
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
      "subtype": SPAN_SUBTYPES.CONTENT_GENERATION,
      "attributes": [
        [
              {
                "_comment": "tool type",
                "attribute": "type",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.get_tool_type(arguments['span'])
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
                "accessor": lambda arguments: _helper.get_source_agent()
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

WORKFLOW = {
    "type": SPAN_TYPES.AGENTIC_REQUEST,
    "subtype": SPAN_SUBTYPES.TURN,
    "is_auto_close": lambda kwargs: False,
    "response_processor": process_workflow_handler,
    "attributes": [
        [
            {
                "_comment": "LlamaIndex event-driven workflow turn",
                "attribute": "type",
                "accessor": lambda arguments: 'workflow.llamaindex'
            },
            {
                "_comment": "workflow class name",
                "attribute": "name",
                "accessor": lambda arguments: type(arguments['instance']).__name__
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "input to the workflow run",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_workflow_input(arguments['args'], arguments['kwargs'])
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "result returned by the workflow run",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_workflow_output(arguments)
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
                "accessor": lambda arguments: _helper.get_source_agent()
              },
              {
                "_comment": "name of the agent called",
                "attribute": "to_agent",
                "accessor": lambda arguments: _helper.get_target_agent(arguments['result']),
                "phase": "post_execution"
              }
        ]
      ]
}