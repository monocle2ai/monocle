from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.copilot_cli.entities.agent import (
    INFERENCE,
    INVOCATION,
    REQUEST,
    SESSION_SUMMARY,
)
from monocle_apptrace.instrumentation.metamodel.copilot_cli.entities.tool import MCP_TOOL, TOOL

_PKG = "monocle_apptrace.instrumentation.metamodel.copilot_cli.replay_handlers"
_OBJ = "ReplayHandler"
_HANDLER = "copilot_handler"

COPILOT_CLI_PROXY_METHODS = [
    {
        "package": _PKG,
        "object": _OBJ,
        "method": "handle_turn",
        "span_name": "Copilot CLI",
        "wrapper_method": task_wrapper,
        "output_processor": REQUEST,
        "span_handler": _HANDLER,
    },
    {
        "package": _PKG,
        "object": _OBJ,
        "method": "handle_invocation",
        "span_name": "Copilot Invocation",
        "wrapper_method": task_wrapper,
        "output_processor": INVOCATION,
        "span_handler": _HANDLER,
    },
    {
        "package": _PKG,
        "object": _OBJ,
        "method": "handle_inference_round",
        "span_name": "Copilot Inference",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE,
        "span_handler": _HANDLER,
    },
    {
        "package": _PKG,
        "object": _OBJ,
        "method": "handle_tool_call",
        "span_name": "Tool",
        "wrapper_method": task_wrapper,
        "output_processor": TOOL,
        "span_handler": _HANDLER,
    },
    {
        "package": _PKG,
        "object": _OBJ,
        "method": "handle_mcp_call",
        "span_name": "MCP Tool",
        "wrapper_method": task_wrapper,
        "output_processor": MCP_TOOL,
        "span_handler": _HANDLER,
    },
    {
        "package": _PKG,
        "object": _OBJ,
        "method": "handle_session_summary",
        "span_name": "Copilot Session Summary",
        "wrapper_method": task_wrapper,
        "output_processor": SESSION_SUMMARY,
        "span_handler": _HANDLER,
    },
]
