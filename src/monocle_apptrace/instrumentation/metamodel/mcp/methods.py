from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper
from monocle_apptrace.instrumentation.metamodel.mcp.entities.inference import (
    TOOLS
)
MCP_METHODS = [
    {
        "package": "mcp.shared.session",
         "object": "BaseSession",
         "method": "send_request",
         "wrapper_method": atask_wrapper,
         "span_handler":"mcp_agent_handler",
         "output_processor": TOOLS
    }
]