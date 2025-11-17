from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper
from monocle_apptrace.instrumentation.metamodel.agents.entities.inference import (
    AGENT,
    AGENT_DELEGATION,
    TOOLS,
    AGENT_REQUEST
)
from monocle_apptrace.instrumentation.metamodel.agents.agents_processor import (
    constructor_wrapper,
    handoff_constructor_wrapper,
)

AGENTS_METHODS = [
    # Main agent runner methods
    {
        "package": "agents.run",
        "object": "Runner",
        "method": "run",
        "wrapper_method": atask_wrapper,
        "span_handler": "agents_agent_handler",
        "output_processor": AGENT_REQUEST,
    },
    {
        "package": "agents.run",
        "object": "Runner",
        "method": "run_sync",
        "wrapper_method": task_wrapper,
        "span_handler": "agents_agent_handler",
        "output_processor": AGENT_REQUEST,
    },
    # AgentRunner class methods (internal runner)
    {
        "package": "agents.run",
        "object": "AgentRunner",
        "method": "_run_single_turn",
        "wrapper_method": atask_wrapper,
        "span_handler": "agents_agent_handler",
        "output_processor": AGENT,
    },
    # Function tool decorator - wrap the function_tool function directly
    {
        "package": "agents.tool",
        "object": "FunctionTool",
        "method": "__init__",  # Empty string means wrap the function itself
        "wrapper_method": constructor_wrapper,
        "output_processor": TOOLS,
    },
]
