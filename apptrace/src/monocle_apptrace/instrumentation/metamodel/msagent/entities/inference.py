"""Inference entity definitions for Microsoft Agent Framework."""

import time
import logging
from types import SimpleNamespace
from monocle_apptrace.instrumentation.common.constants import (
    SPAN_SUBTYPES,
    SPAN_TYPES,
)
from monocle_apptrace.instrumentation.metamodel.msagent import _helper
from monocle_apptrace.instrumentation.common.utils import get_error_message, patch_instance_method, resolve_from_alias

logger = logging.getLogger(__name__)

# For Microsoft Agent Framework, turn doesn't include agent name (follows ADK pattern)
AGENT_REQUEST = {
    "type": SPAN_TYPES.AGENTIC_REQUEST,
    "subtype": SPAN_SUBTYPES.TURN,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.microsoft"
            },
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is Agent input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_request_agent_input(arguments)
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is response from Agent",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(arguments["result"])
                }
            ]
        }
    ]
}

# Agent invocation span (agentic.invocation with content_processing subtype)
# Used for chat client get_response/get_streaming_response
AGENT = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.microsoft",
            },
            {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_chat_client_name(arguments["instance"]),
            },
            {
                "_comment": "model id",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_chat_client_model(arguments["instance"]),
            },
            {
                "_comment": "delegating agent name",
                "attribute": "from_agent",
                "accessor": lambda arguments: _helper.get_from_agent_name(arguments)
            },
            {
                "_comment": "from_agent invocation id",
                "attribute": "from_agent_span_id",
                "accessor": lambda arguments: _helper.get_from_agent_span_id(arguments)
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
                    "accessor": lambda arguments: _helper.extract_chat_client_input(arguments),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
                {
                    "_comment": "this is response from Agent",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_chat_client_response(
                        arguments["result"], arguments.get("span")
                    ),
                },
            ],
        },
    ],
}


# Workflow turn span (agentic.turn with turn subtype) - no input/output events
WORKFLOW_TURN = {
    "type": SPAN_TYPES.AGENTIC_REQUEST,
    "subtype": SPAN_SUBTYPES.TURN,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.microsoft"
            },
        ],
    ],
    "events": []
}

# ChatAgent invocation span (agentic.invocation with content_processing subtype)
CHAT_AGENT_INVOCATION = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.microsoft",
            },
            {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_agent_name(arguments["instance"]),
            },
            {
                "_comment": "model/description",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_chat_client_model(arguments["instance"]),
            },
            {
                "_comment": "delegating agent name",
                "attribute": "from_agent",
                "accessor": lambda arguments: _helper.get_from_agent_name(arguments)
            },
            {
                "_comment": "from_agent invocation id",
                "attribute": "from_agent_span_id",
                "accessor": lambda arguments: _helper.get_from_agent_span_id(arguments)
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
                    "accessor": lambda arguments: _helper.extract_request_agent_input(arguments)
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is response from Agent",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(arguments["result"])
                }
            ]
        }
    ]
}

TOOL = {
    "type": SPAN_TYPES.AGENTIC_TOOL_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_GENERATION,
    "attributes": [
        [
            {
                "_comment": "tool type",
                "attribute": "type",
                "accessor": lambda arguments: "tool.microsoft",
            },
            {
                "_comment": "name of the tool",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_tool_name(arguments["instance"]),
            },
            {
                "_comment": "tool description",
                "attribute": "description",
                "accessor": lambda arguments: _helper.get_tool_description(
                    arguments["instance"]
                ),
            },
        ],
        [
            {
                "_comment": "agent name (owner of the tool)",
                "attribute": "name",
                "accessor": lambda arguments: _helper.get_agent_name_from_context(),
            },
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.microsoft",
            },
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is Tool input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_tool_input(arguments),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
                {
                    "_comment": "this is response from Tool",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_tool_response(
                        arguments["result"], arguments.get("span")
                    ),
                },
            ],
        },
    ],
}


INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "subtype": lambda arguments: _helper.agent_inference_type(arguments),
    "attributes": [
        [
            {
                "_comment": "provider type ,name , deployment , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: "inference."
                + (_helper.get_inference_type(arguments["instance"]))
                or "openai",
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: _helper.extract_provider_name(
                    arguments["instance"]
                ),
            },
            {
                "attribute": "deployment",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["instance"].__dict__,
                    [
                        "engine",
                        "azure_deployment",
                        "deployment_name",
                        "deployment_id",
                        "deployment",
                    ],
                ),
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: resolve_from_alias(
                    arguments["instance"].__dict__,
                    ["azure_endpoint", "api_base", "endpoint"],
                )
                or _helper.extract_inference_endpoint(arguments["instance"]),
            },
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: _helper.extract_model_name(
                    arguments["instance"], arguments["kwargs"]
                ),
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: _helper.extract_model_type(
                    arguments["instance"], arguments["kwargs"]
                ),
            },
        ],
        [
            {
                "_comment": "Tool name when finish_type is tool_call",
                "attribute": "name",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.extract_tool_name(arguments),
            },
            {
                "_comment": "Tool type when finish_type is tool_call", 
                "attribute": "type",
                "phase": "post_execution",
                "accessor": lambda arguments: _helper.extract_tool_type(arguments),
            },
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is instruction and user query to LLM",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_messages(
                        arguments["kwargs"]
                    ),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
                {
                    "_comment": "this is result from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(
                        arguments,
                    ),
                },
            ],
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM",
                    "accessor": lambda arguments: _helper.update_span_from_llm_response(
                        arguments["result"]
                    ),
                },
                {
                    "_comment": "finish reason from OpenAI response",
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: _helper.extract_finish_reason(
                        arguments
                    ),
                },
                {
                    "_comment": "finish type mapped from finish reason",
                    "attribute": "finish_type",
                    "accessor": lambda arguments: _helper.map_finish_reason_to_finish_type(
                        _helper.extract_finish_reason(arguments)
                    ),
                }
            ],
        },
    ],
}

