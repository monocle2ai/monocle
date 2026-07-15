from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.metamodel.mcp import _helper
from monocle_apptrace.instrumentation.metamodel.mcp.entities.inference import TOOLS


MCP_METHODS = [
    {
        "package": "mcp.shared.session",
        "object": "BaseSession",
        "method": "send_request",
        "wrapper_method": atask_wrapper,
        "span_handler": "mcp_agent_handler",
        "output_processor": TOOLS,
    },
    {
        "package": "mcp.client.session",
        "object": "ClientSession",
        "method": "initialize",
        "wrapper_method": _helper.mcp_initialize_wrapper,
    },
    {
        "package": "langchain_mcp_adapters.tools",
        "object": "",
        "method": "convert_mcp_tool_to_langchain_tool",
        "wrapper_method": _helper.langchain_mcp_wrapper,
    }
]
