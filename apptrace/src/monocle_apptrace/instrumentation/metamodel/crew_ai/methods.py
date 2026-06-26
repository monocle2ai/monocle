from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper
from monocle_apptrace.instrumentation.metamodel.crew_ai.entities.inference import (
    AGENT,
    TOOLS,
)

CREW_AI_METHODS = [
    # Core CrewAI orchestration
    {
        "package": "crewai.crew",
        "object": "Crew",
        "method": "kickoff",
        "wrapper_method": task_wrapper,
        "span_handler": "crew_ai_agent_handler",
        "output_processor": AGENT,
    },
    {
        "package": "crewai.crew",
        "object": "Crew",
        "method": "kickoff_async",
        "wrapper_method": atask_wrapper,
        "span_handler": "crew_ai_agent_handler",
        "output_processor": AGENT,
    },
    {
        "package": "crewai.agent",
        "object": "Agent",
        "method": "execute_task",
        "wrapper_method": task_wrapper,
        "span_handler": "crew_ai_agent_handler",
        "output_processor": AGENT,
    },

    # Task execution
    {
        "package": "crewai.task",
        "object": "Task",
        "method": "execute_sync",
        "wrapper_method": task_wrapper,
        "span_handler": "crew_ai_task_handler",
        "output_processor": AGENT,
    },
    {
        # execute_async is a SYNC method returning a concurrent.futures.Future (it spawns a
        # thread), not a coroutine — the async atask_wrapper made it a coroutine and crashed
        # future.result(). Use task_wrapper; no span here (skip_span), just context capture.
        "package": "crewai.task",
        "object": "Task",
        "method": "execute_async",
        "wrapper_method": task_wrapper,
        "span_handler": "crew_ai_task_handler",
        "output_processor": AGENT,
    },
    {
        # Worker-thread entry point for async tasks. No span; the handler re-attaches the
        # context captured in execute_async so child spans share the turn's trace + session.
        "package": "crewai.task",
        "object": "Task",
        "method": "_execute_task_async",
        "wrapper_method": task_wrapper,
        "span_handler": "crew_ai_task_handler",
        "output_processor": AGENT,
    },

    # Tool execution - public interface
    {
        "package": "crewai.tools.base_tool",
        "object": "BaseTool", 
        "method": "run",
        "wrapper_method": task_wrapper,
        "span_handler": "crew_ai_tool_handler",
        "output_processor": TOOLS,
    },
    # Tool execution - implementation method
    {
        "package": "crewai.tools.base_tool",
        "object": "BaseTool", 
        "method": "_run",
        "wrapper_method": task_wrapper,
        "span_handler": "crew_ai_tool_handler",
        "output_processor": TOOLS,
    },
    {
        "package": "crewai.tools.structured_tool",
        "object": "CrewStructuredTool",
        "method": "invoke",
        "wrapper_method": task_wrapper,
        "span_handler": "crew_ai_tool_handler",
        "output_processor": TOOLS,
    },
]