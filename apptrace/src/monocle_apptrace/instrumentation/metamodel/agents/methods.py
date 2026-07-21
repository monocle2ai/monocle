from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper
from monocle_apptrace.instrumentation.metamodel.agents.entities.inference import (
    AGENT,
    AGENT_DELEGATION,
    TOOLS,
    AGENT_REQUEST,
    AGENT_REQUEST_STREAM,
    AGENT_STREAM,
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
    # Streaming entrypoint. run_streamed returns synchronously with a
    # RunResultStreaming; the turn + invocation spans stay open (is_auto_close=False)
    # and are finalized by process_agent_stream once the caller consumes the stream.
    {
        "package": "agents.run",
        "object": "Runner",
        "method": "run_streamed",
        "wrapper_method": task_wrapper,
        "span_handler": "agents_agent_handler",
        "output_processor_list": [AGENT_REQUEST_STREAM, AGENT_STREAM],
    },
    # Per-agent invocation. openai-agents>=0.16 moved this from
    # AgentRunner._run_single_turn to the module-level agents.run.run_single_turn.
    # Register both targets so instrumentation works across SDK versions; the target that
    # does not exist in the installed version is skipped (logged) rather than fatal.
    {
        "package": "agents.run",
        "object": None,  # module-level function; span_name set since the default joins package.object.method
        "method": "run_single_turn",
        "span_name": "agents.run.run_single_turn",
        "wrapper_method": atask_wrapper,
        "span_handler": "agents_agent_handler",
        "output_processor": AGENT,
    },
    {
        # openai-agents<0.16: AgentRunner._run_single_turn (class method).
        "package": "agents.run",
        "object": "AgentRunner",
        "method": "_run_single_turn",
        "span_name": "agents.run.run_single_turn",
        "wrapper_method": atask_wrapper,
        "span_handler": "agents_agent_handler",
        "output_processor": AGENT,
    },
    # Streaming per-turn internal. Defined and called within agents.run_internal.run_loop
    # (not the agents.run namespace). skip_span drops this when it runs under run_streamed,
    # since run_streamed already emits the invocation span; kept registered to cover any
    # direct call path.
    {
        "package": "agents.run_internal.run_loop",
        "object": "",
        "method": "run_single_turn_streamed",
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
    {
        "package": "agents.tool",
        "object": "",
        "method": "invoke_function_tool",
        "wrapper_method": atask_wrapper,
        "span_handler": "agents_agent_handler",
        "output_processor": TOOLS,
    },
]
