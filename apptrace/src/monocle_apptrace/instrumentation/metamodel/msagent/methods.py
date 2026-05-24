"""Microsoft Agent Framework method definitions for instrumentation.

"""

from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.metamodel.msagent.entities.inference import (
    AGENT, 
    AGENT_REQUEST,
    INFERENCE,
    TOOL,
)


MSAGENT_METHODS = [
    # Workflow-level methods - top level span for multi-agent workflows
    {
        "package": "agent_framework._workflows._workflow",
        "object": "Workflow",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    # Agent.run - agent request level (replaces ChatAgent.run)
    # Note: Agent.run has stream parameter but returns different types
    # For now, instrument as non-streaming; streaming may need separate handling
    {
        "package": "agent_framework._agents",
        "object": "Agent",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    # BaseChatClient - invocation level (base class for all chat clients)
    {
        "package": "agent_framework._clients",
        "object": "BaseChatClient",
        "method": "get_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT,
    },
    # BaseChatClient inference-level - the actual LLM API call
    {
        "package": "agent_framework._clients",
        "object": "BaseChatClient",
        "method": "_inner_get_response",
        "span_handler": "msagent_inference_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE,
    },
    # AgentExecutor - agent invocation within workflow
    {
        "package": "agent_framework._workflows._agent_executor",
        "object": "AgentExecutor",
        "method": "execute",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT,
    },
    # FunctionExecutor - tool invocation
    {
        "package": "agent_framework._workflows._function_executor",
        "object": "FunctionExecutor",
        "method": "execute",
        "wrapper_method": atask_wrapper,
        "output_processor": TOOL,
    },
]