from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.langgraph.entities.inference import (
    AGENT, TOOLS
)
LANGGRAPH_METHODS = [
#TODO: Add async methods
    {
        "package": "langgraph.graph.state",
         "object": "CompiledStateGraph",
         "method": "invoke",
         "wrapper_method": task_wrapper,
         "output_processor": AGENT
    },
    {
        "package": "langchain_core.tools.base",
         "object": "BaseTool",
         "method": "run",
         "wrapper_method": task_wrapper,
         "output_processor": TOOLS
    }
]