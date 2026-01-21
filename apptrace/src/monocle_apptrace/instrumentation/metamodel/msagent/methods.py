"""Microsoft Agent Framework method definitions for instrumentation."""

# import inspect
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, atask_iter_wrapper
from monocle_apptrace.instrumentation.metamodel.msagent.entities.inference import (
    AGENT, 
    AGENT_REQUEST, 
    TOOL,
)

# def should_skip_chat_client_method(instance, *args, **kwargs):
#     """Skip instrumentation if called from ChatAgent.run or ChatAgent.run_stream."""
#     frame = inspect.currentframe()
#     try:
#         # Walk up the call stack
#         while frame:
#             frame_info = inspect.getframeinfo(frame)
#             code_context = frame.f_code
            
#             # Check if we're in ChatAgent.run or ChatAgent.run_stream
#             if (code_context.co_name in ['run', 'run_stream'] and 
#                 'agent_framework' in frame_info.filename and
#                 '_agents.py' in frame_info.filename):
#                 return True
            
#             frame = frame.f_back
#     finally:
#         del frame
    
#     return False


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
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework.azure._chat_client",
        "object": "AzureOpenAIChatClient",
        "method": "get_streaming_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": AGENT,
        # "should_skip": should_skip_chat_client_method,
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

]


