"""Microsoft Agent Framework method definitions for instrumentation."""

# import inspect
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, atask_iter_wrapper
from monocle_apptrace.instrumentation.metamodel.msagent.entities.inference import (
    AGENT, 
    AGENT_REQUEST,
    INFERENCE,
    TOOL,
)


MSAGENT_METHODS = [
    # Workflow-level methods - top level span for multi-agent workflows
    {
        "package": "agent_framework._workflows._handoff",
        "object": "Workflow",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    {
        "package": "agent_framework",
        "object": "ChatAgent",
        "method": "run_stream",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    {
        "package": "agent_framework",
        "object": "ChatAgent",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    {
        "package": "agent_framework.azure._chat_client",
        "object": "AzureOpenAIChatClient",
        "method": "get_streaming_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework.azure._chat_client",
        "object": "AzureOpenAIChatClient",
        "method": "get_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework.azure._assistants_client",
        "object": "AzureOpenAIAssistantsClient",
        "method": "get_streaming_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework.azure._assistants_client",
        "object": "AzureOpenAIAssistantsClient",
        "method": "get_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework._tools",
        "object": "AIFunction",
        "method": "invoke",
        "wrapper_method": atask_wrapper,
        "output_processor": TOOL,
    },
        {
        "package": "agent_framework.azure._assistants_client",
        "object": "AzureOpenAIAssistantsClient",
        "method": "_inner_get_response",
        "span_handler": "msagent_inference_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE,
    },

]
