from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper, atask_iter_wrapper
from monocle_apptrace.instrumentation.metamodel.adk.entities.agent import (
    AGENT,
)
from monocle_apptrace.instrumentation.metamodel.adk.entities.tool import (
    TOOL
)

ADK_METHODS = [
    {
      "package": "google.adk.agents.base_agent",
      "object": "BaseAgent",
      "method": "run_async",
      "wrapper_method": atask_iter_wrapper,
      "output_processor": AGENT,
    },
    {
      "package": "google.adk.tools.function_tool",
      "object": "FunctionTool",
      "method": "run_async",
      "wrapper_method": atask_wrapper,
      "output_processor": TOOL,
    }
]