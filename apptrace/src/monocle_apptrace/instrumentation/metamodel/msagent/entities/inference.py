"""Inference entity definitions for Microsoft Agent Framework."""

import time
import logging
from types import SimpleNamespace
from monocle_apptrace.instrumentation.common.constants import (
    SPAN_SUBTYPES,
    SPAN_TYPES,
)
from monocle_apptrace.instrumentation.metamodel.msagent import _helper
from monocle_apptrace.instrumentation.common.utils import get_error_message, patch_instance_method

logger = logging.getLogger(__name__)


def process_msagent_stream(to_wrap, response, span_processor):
    """
    Process Microsoft Agent Framework streaming responses.
    Patches __anext__ to accumulate streaming chunks and capture final response.
    Similar to Azure AI Inference's process_stream but adapted for Microsoft Agent Framework.
    """
    waiting_for_first_token = True
    stream_start_time = time.time_ns()
    first_token_time = stream_start_time
    stream_closed_time = None
    accumulated_response = ""
    tools = None
    role = "assistant"
    
    # Microsoft Agent Framework uses async iterators - patch __anext__
    if to_wrap and hasattr(response, "__anext__"):
        original_anext = response.__anext__
        
        async def new_anext(self):
            nonlocal waiting_for_first_token, first_token_time, stream_closed_time, accumulated_response, tools, role
            
            try:
                item = await original_anext()
                
                # Handle Microsoft Agent Framework streaming chunks (AgentRunResponseUpdate or ChatResponseUpdate)
                # Check for first token
                if waiting_for_first_token:
                    waiting_for_first_token = False
                    first_token_time = time.time_ns()
                
                # Accumulate content from the stream item - both have 'text' attribute
                if hasattr(item, "text"):
                    text = item.text
                    if text:
                        accumulated_response += str(text)
                
                return item
                
            except StopAsyncIteration:
                # Stream is complete, process final span
                stream_closed_time = time.time_ns()
                
                if span_processor:
                    ret_val = SimpleNamespace(
                        type="stream",
                        timestamps={
                            "data.input": int(stream_start_time),
                            "data.output": int(first_token_time),
                            "metadata": int(stream_closed_time or time.time_ns()),
                        },
                        output_text=accumulated_response,
                        tools=tools,
                        role=role,
                    )
                    span_processor(ret_val)
                raise
            except Exception as e:
                logger.warning(
                    "Warning: Error occurred while processing MS Agent stream item: %s",
                    str(e),
                )
                raise
        
        patch_instance_method(response, "__anext__", new_anext)

# Turn-level request span (agentic.request with turn subtype)
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
