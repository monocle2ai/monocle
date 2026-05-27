"""Microsoft Agent Framework method definitions for instrumentation.

"""

from monocle_apptrace.instrumentation.common.wrapper import (
    atask_wrapper, 
    atask_iter_wrapper,
    amonocle_wrapper,
    amonocle_iter_wrapper,
    with_tracer_wrapper
)
from opentelemetry.trace import Tracer
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.msagent.entities.inference import (
    AGENT, 
    AGENT_REQUEST,
    TOOL,
)


def msagent_adaptive_wrapper_dispatch(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    """
    Dispatch wrapper that routes to appropriate wrapper based on context.
    The method signatures and return types are compatible across these versions.
    """
    # Check if inside workout context
    if handler.skip_span(to_wrap, wrapped, instance, args, kwargs):
        return wrapped(*args, **kwargs)
    
    # Standalone agent call - check if streaming
    if kwargs.get("stream", False):
        # Streaming mode - return async generator
        return amonocle_iter_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
    else:
        # Non-streaming mode - return coroutine
        return amonocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)


# Wrap with tracer
msagent_adaptive_wrapper = with_tracer_wrapper(msagent_adaptive_wrapper_dispatch)


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
    # Agent.run - agent request level (standalone agent calls with streaming support)
    {
        "package": "agent_framework._agents",
        "object": "Agent",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": msagent_adaptive_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    # NOTE: BaseChatClient.get_response and _inner_get_response are NOT instrumented
    # because they break when wrapped (SDK internal code expects specific return types).
    # Streaming details are captured via the stream processor on Agent.run instead.
    
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
        "span_handler": "msagent_tool_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": TOOL,
    },
]