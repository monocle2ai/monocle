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


@with_tracer_wrapper
async def msagent_adaptive_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    """
    Adaptive wrapper for MS Agent methods that can return either a single result or an async iterator.
    When stream=True, it yields multiple items. When stream=False, it awaits and yields once.
    """
    if kwargs.get("stream", False):
        async for item in amonocle_iter_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs):
            yield item
    else:
        result = await amonocle_wrapper(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
        yield result


MSAGENT_METHODS = [
    # Workflow-level methods - top level span for multi-agent workflows
    {
        "package": "agent_framework._workflows._workflow",
        "object": "Workflow",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    # Agent.run - agent request level (user-facing, needs adaptive wrapper for stream support)
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
        "wrapper_method": atask_wrapper,
        "output_processor": TOOL,
    },
]