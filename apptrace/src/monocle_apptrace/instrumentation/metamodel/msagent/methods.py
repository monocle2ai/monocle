"""Microsoft Agent Framework method definitions for instrumentation."""

from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper, atask_iter_wrapper
from monocle_apptrace.instrumentation.metamodel.msagent.entities.inference import AGENT, AGENT_ORCHESTRATOR, AGENT_REQUEST, TOOL


def accumulate_streaming_response(to_wrap, response, callback):
    """
    Response processor for streaming agent responses.
    Accumulates text from streaming chunks to capture final response.
    """
    accumulated_text = ""
    if hasattr(response, "text") and response.text:
        accumulated_text = response.text
    elif hasattr(response, "content") and response.content:
        accumulated_text = response.content
    elif isinstance(response, str):
        accumulated_text = response
    
    # Store accumulated text for later retrieval
    if accumulated_text:
        if not hasattr(to_wrap, "_accumulated_response"):
            to_wrap["_accumulated_response"] = ""
        to_wrap["_accumulated_response"] += accumulated_text
    
    callback(to_wrap["_accumulated_response"] if "_accumulated_response" in to_wrap else response)


MSAGENT_METHODS = [
    # Workflow-level methods (agentic.request/turn) - top level span for multi-agent workflows
    # Workflow.run_stream creates the single top-level agentic.turn span
    {
        "package": "agent_framework._workflows",
        "object": "Workflow",
        "method": "run_stream",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": {
            **AGENT_ORCHESTRATOR,
            "response_processor": accumulate_streaming_response
        },
    },
    # Workflow.run creates the single top-level agentic.turn span
    {
        "package": "agent_framework._workflows",
        "object": "Workflow",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT_ORCHESTRATOR,
    },
    
    # Turn-level methods (agentic.request/turn) - top level span for single agent
    # ChatAgent.run_stream creates agentic.request span (turn) only when NOT in workflow context
    {
        "package": "agent_framework",
        "object": "ChatAgent",
        "method": "run_stream",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": {
            **AGENT_REQUEST,
            "response_processor": accumulate_streaming_response
        },
    },
    # ChatAgent.run creates agentic.request span (turn) only when NOT in workflow context
    {
        "package": "agent_framework",
        "object": "ChatAgent",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": task_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    
    # Agent invocation methods (agentic.invocation) - nested under turn
    # Chat client get_streaming_response creates agentic.invocation span
    # Instrument both Azure and base implementations
    {
        "package": "agent_framework.azure._chat_client",
        "object": "AzureOpenAIChatClient",
        "method": "get_streaming_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": {
            **AGENT,
            "response_processor": accumulate_streaming_response
        },
    },
    {
        "package": "agent_framework._clients",
        "object": "BaseChatClient",
        "method": "get_streaming_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": {
            **AGENT,
            "response_processor": accumulate_streaming_response
        },
    },
    # Chat client get_response creates agentic.invocation span
    {
        "package": "agent_framework.azure._chat_client",
        "object": "AzureOpenAIChatClient",
        "method": "get_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework._clients",
        "object": "BaseChatClient",
        "method": "get_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT,
    },
    
    # Tool invocation methods (agentic tool invocation)
    # AIFunction.invoke is the method that executes Python functions wrapped as tools
    # Note: invoke is an async method, so we use atask_wrapper
    {
        "package": "agent_framework._tools",
        "object": "AIFunction",
        "method": "invoke",
        "wrapper_method": atask_wrapper,
        "span_handler": "msagent_tool_handler",
        "output_processor": TOOL,
    },
]

