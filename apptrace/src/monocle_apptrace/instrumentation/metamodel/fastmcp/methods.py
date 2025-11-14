from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.metamodel.fastmcp.entities.tools import TOOLS, LIST_TOOLS
from monocle_apptrace.instrumentation.metamodel.fastmcp.entities.resources import RESOURCES, LIST_RESOURCES
from monocle_apptrace.instrumentation.metamodel.fastmcp.entities.prompts import PROMPTS, LIST_PROMPTS


FASTMCP_METHODS = [
    {
        "package": "fastmcp.server.server",
        "object": "FastMCP",
        "method": "_call_tool_mcp",
        "wrapper_method": atask_wrapper,
        "output_processor": TOOLS,
        "span_name": "fastmcp.tool_call"
    },
    {
        "package": "fastmcp.server.server",
        "object": "FastMCP",
        "method": "_read_resource_mcp",
        "wrapper_method": atask_wrapper,
        "output_processor": RESOURCES,
        "span_name": "fastmcp.resource_read"
    },
    {
        "package": "fastmcp.server.server",
        "object": "FastMCP",
        "method": "_get_prompt_mcp",
        "wrapper_method": atask_wrapper,
        "output_processor": PROMPTS,
        "span_name": "fastmcp.prompt_get"
    },
    {
        "package": "fastmcp.server.server",
        "object": "FastMCP",
        "method": "_list_tools_mcp",
        "wrapper_method": atask_wrapper,
        "output_processor": LIST_TOOLS,
        "span_name": "fastmcp.tools_list"
    },
    {
        "package": "fastmcp.server.server",
        "object": "FastMCP",
        "method": "_list_resources_mcp",
        "wrapper_method": atask_wrapper,
        "output_processor": LIST_RESOURCES,
        "span_name": "fastmcp.resources_list"
    },
    {
        "package": "fastmcp.server.server",
        "object": "FastMCP",
        "method": "_list_prompts_mcp",
        "wrapper_method": atask_wrapper,
        "output_processor": LIST_PROMPTS,
        "span_name": "fastmcp.prompts_list"
    },
]
