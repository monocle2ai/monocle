import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DESCRIPTION_FIELDS = ("description", "command", "query", "url", "file_path", "pattern", "path")


def extract_agent_request_input(arguments) -> str:
    return arguments["kwargs"].get("prompt", "")


def extract_agent_response(result) -> str:
    if isinstance(result, str):
        return result
    return str(result) if result else ""


def get_tool_type(arguments) -> str:
    tool_name = arguments["kwargs"].get("tool_name", "")
    if tool_name.startswith("mcp__"):
        return "tool.mcp"
    return "tool.codex_cli"


def get_tool_name(arguments) -> str:
    return arguments["kwargs"].get("tool_name", "")


def get_tool_description(arguments) -> str:
    tool_name = arguments["kwargs"].get("tool_name", "")
    tool_input = arguments["kwargs"].get("tool_input", {})
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__", 2)
        return f"{parts[1]} / {parts[2]}" if len(parts) == 3 else tool_name
    if isinstance(tool_input, dict):
        for field in _DESCRIPTION_FIELDS:
            val = tool_input.get(field)
            if val and isinstance(val, str):
                return val[:120]
    if isinstance(tool_input, str) and tool_input:
        return tool_input[:120]
    return tool_name


def extract_tool_input(arguments) -> str:
    tool_input = arguments["kwargs"].get("tool_input", {})
    if isinstance(tool_input, (dict, list)):
        return json.dumps(tool_input)
    return str(tool_input) if tool_input else ""


def extract_tool_response(result) -> str:
    return str(result) if result else ""


def find_subagent_transcript(parent_transcript_path: str, thread_id: str):
    """Locate ~/.codex/sessions/.../rollout-*-<thread_id>.jsonl."""
    if not thread_id:
        return None
    parent = Path(parent_transcript_path) if parent_transcript_path else None
    candidates = []
    if parent and parent.parent.exists():
        candidates = list(parent.parent.glob(f"rollout-*-{thread_id}.jsonl"))
    if not candidates:
        sessions_root = Path.home() / ".codex" / "sessions"
        if sessions_root.exists():
            candidates = list(sessions_root.glob(f"**/rollout-*-{thread_id}.jsonl"))
    return candidates[0] if candidates else None
