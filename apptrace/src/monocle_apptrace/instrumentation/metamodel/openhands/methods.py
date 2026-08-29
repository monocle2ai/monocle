from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.openhands.entities.agent import AGENT, AGENT_REQUEST
from monocle_apptrace.instrumentation.metamodel.openhands.entities.tool import TOOL

OPENHANDS_METHODS = [
    {
        "package": "openhands.sdk.conversation.impl.local_conversation",
        "object": "LocalConversation",
        "method": "run",
        "wrapper_method": task_wrapper,
        "span_handler": "openhands_handler",
        "output_processor": AGENT_REQUEST
    },
    {
        "package": "openhands.sdk.conversation.impl.local_conversation",
        "object": "LocalConversation",
        "method": "arun",
        "wrapper_method": atask_wrapper,
        "span_handler": "openhands_handler",
        "output_processor": AGENT_REQUEST
    },
    {
        "package": "openhands.sdk.agent.agent",
        "object": "Agent",
        "method": "step",
        "wrapper_method": task_wrapper,
        "output_processor": AGENT
    },
    {
        "package": "openhands.sdk.agent.agent",
        "object": "Agent",
        "method": "astep",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT
    },
    {
        # Stash-only boundary: captures the caller's context per action before
        # the sync parallel path fans out to threads. Never emits a span
        # (handler skip), so no output_processor.
        "package": "openhands.sdk.agent.parallel_executor",
        "object": "ParallelToolExecutor",
        "method": "execute_batch",
        "wrapper_method": task_wrapper,
        "span_handler": "openhands_tool_handler"
    },
    {
        # Sync per-tool boundary; runs on the calling thread in the sequential
        # config, on a worker thread (with restored context) in parallel mode.
        "package": "openhands.sdk.agent.parallel_executor",
        "object": "ParallelToolExecutor",
        "method": "_run_safe",
        "wrapper_method": task_wrapper,
        "span_handler": "openhands_tool_handler",
        "output_processor": TOOL
    },
    {
        # Async per-tool boundary; a coroutine on the event-loop thread, so the
        # span parents correctly even when tool calls fan out to worker threads.
        "package": "openhands.sdk.agent.parallel_executor",
        "object": "ParallelToolExecutor",
        "method": "_arun_safe",
        "wrapper_method": atask_wrapper,
        "span_handler": "openhands_tool_handler",
        "output_processor": TOOL
    }
]
