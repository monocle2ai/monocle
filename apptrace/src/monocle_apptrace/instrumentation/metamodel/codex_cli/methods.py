from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.codex_cli.entities.agent import (
    INFERENCE,
    INVOCATION,
    REQUEST,
    SUBAGENT_INVOCATION,
)
from monocle_apptrace.instrumentation.metamodel.codex_cli.entities.tool import MCP_TOOL, TOOL

_PKG = "monocle_apptrace.instrumentation.metamodel.codex_cli.replay_handlers"
_OBJ = "ReplayHandler"
_HANDLER = "codex_handler"

CODEX_CLI_PROXY_METHODS = [
    {
        "package": _PKG,
        "object": _OBJ,
        "method": "handle_turn",
        "span_name": "Codex CLI",
        "wrapper_method": task_wrapper,
        "output_processor": REQUEST,
        "span_handler": _HANDLER,
    },
    {
        "package": _PKG,
        "object": _OBJ,
        "method": "handle_invocation",
        "span_name": "Codex Invocation",
        "wrapper_method": task_wrapper,
        "output_processor": INVOCATION,
        "span_handler": _HANDLER,
    },
    {
        "package": _PKG,
        "object": _OBJ,
        "method": "handle_inference_round",
        "span_name": "Codex Inference",
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
        "method": "handle_subagent",
        "span_name": "Sub-Agent",
        "wrapper_method": task_wrapper,
        "output_processor": SUBAGENT_INVOCATION,
        "span_handler": _HANDLER,
    },
]
