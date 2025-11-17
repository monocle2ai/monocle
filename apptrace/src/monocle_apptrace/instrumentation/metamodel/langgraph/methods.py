from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper
from monocle_apptrace.instrumentation.metamodel.langgraph.entities.inference import (
    AGENT,
    TOOLS,
)

LANGGRAPH_METHODS = [
    {
        "package": "langgraph.graph.state",
        "object": "CompiledStateGraph",
        "method": "invoke",
        "wrapper_method": task_wrapper,
        "span_handler": "langgraph_agent_handler",
        "scope_name": "agent.invocation",
        "output_processor": AGENT,
    },
    {
        "package": "langgraph.graph.state",
        "object": "CompiledStateGraph",
        "method": "ainvoke",
        "wrapper_method": atask_wrapper,
        "span_handler": "langgraph_agent_handler",
        "scope_name": "agent.invocation",
        "output_processor": AGENT,
    },
    {
        "package": "langchain_core.tools.simple",
        "object": "Tool",
        "method": "_run",
        "wrapper_method": task_wrapper,
        "span_handler": "langgraph_tool_handler",
        "output_processor": TOOLS,
    },
    {
        "package": "langchain_core.tools.structured",
        "object": "StructuredTool",
        "method": "_run",
        "wrapper_method": task_wrapper,
        "span_handler": "langgraph_tool_handler",
        "output_processor": TOOLS,
    },
    {
        "package": "langchain_core.tools.simple",
        "object": "Tool",
        "method": "_arun",
        "wrapper_method": atask_wrapper,
        "span_handler": "langgraph_tool_handler",
        "output_processor": TOOLS,
    },
    {
        "package": "langchain_core.tools.structured",
        "object": "StructuredTool",
        "method": "_arun",
        "wrapper_method": atask_wrapper,
        "span_handler": "langgraph_tool_handler",
        "output_processor": TOOLS,
    },
]
