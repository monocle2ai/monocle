import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

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
    return "tool.claude_cli"

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
    return tool_name

def extract_tool_input(arguments) -> str:
    tool_input = arguments["kwargs"].get("tool_input", {})
    if isinstance(tool_input, dict):
        return json.dumps(tool_input)
    return str(tool_input) if tool_input else ""

def extract_tool_response(result) -> str:
    if isinstance(result, dict):
        if "stdout" in result:
            return result["stdout"]
        return json.dumps(result)
    return str(result) if result else ""

def read_transcript_tokens(transcript_path: str, start_line: int = 0) -> Dict[str, int]:
    """Sum assistant message usage from transcript lines at or after start_line.

    prompt_tokens = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
    Raw input_tokens alone is wrong — Claude Code caches aggressively, making
    it as small as 3 on a warm session.

    start_line allows callers to pass the line count from the previous turn so
    only the current turn's API calls are counted (avoids double-counting cached
    tokens across turns).
    """
    if not transcript_path:
        return {}
    path = Path(transcript_path)
    if not path.exists():
        return {}

    total_input = total_cache_read = total_cache_creation = total_output = 0
    seen_request_ids: set = set()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[start_line:]:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                inner = msg.get("message", {})
                if not isinstance(inner, dict) or inner.get("role") != "assistant":
                    continue
                req_id = msg.get("requestId", "")
                if req_id:
                    if req_id in seen_request_ids:
                        continue
                    seen_request_ids.add(req_id)
                usage = inner.get("usage", {})
                total_input += usage.get("input_tokens", 0)
                total_cache_read += usage.get("cache_read_input_tokens", 0)
                total_cache_creation += usage.get("cache_creation_input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
            except Exception as e:
                logger.debug(f"skipping transcript line: {e}")
                continue
    except Exception as e:
        logger.debug(f"Error reading transcript {transcript_path}: {e}")
        return {}

    prompt_t = total_input + total_cache_read + total_cache_creation
    if not prompt_t and not total_output:
        return {}

    return {
        "prompt_tokens": prompt_t,
        "completion_tokens": total_output,
        "total_tokens": prompt_t + total_output,
        "input_tokens": total_input,
        "cache_read_tokens": total_cache_read,
        "cache_creation_tokens": total_cache_creation,
    }

def read_subagent_transcript(transcript_path: str) -> Tuple[str, Dict[str, int]]:
    """Read model name and token totals from a subagent transcript JSONL.

    Returns (model_name, tokens_dict). Falls back to ("claude", {}) on any failure.
    """
    if not transcript_path:
        return "claude", {}
    path = Path(transcript_path)
    if not path.exists():
        return "claude", {}

    model = "claude"
    total_input = total_cache_read = total_cache_creation = total_output = 0
    seen_request_ids: set = set()
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                inner = msg.get("message", {})
                if not isinstance(inner, dict) or inner.get("role") != "assistant":
                    continue
                req_id = msg.get("requestId", "")
                if req_id:
                    if req_id in seen_request_ids:
                        continue
                    seen_request_ids.add(req_id)
                if model == "claude" and inner.get("model"):
                    model = inner["model"]
                usage = inner.get("usage", {})
                total_input += usage.get("input_tokens", 0)
                total_cache_read += usage.get("cache_read_input_tokens", 0)
                total_cache_creation += usage.get("cache_creation_input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
            except Exception as e:
                logger.debug(f"skipping transcript line: {e}")
                continue
    except Exception as e:
        logger.debug(f"Error reading subagent transcript {transcript_path}: {e}")
        return "claude", {}

    prompt_t = total_input + total_cache_read + total_cache_creation
    if not prompt_t and not total_output:
        return model, {}

    return model, {
        "prompt_tokens": prompt_t,
        "completion_tokens": total_output,
        "total_tokens": prompt_t + total_output,
        "input_tokens": total_input,
        "cache_read_tokens": total_cache_read,
        "cache_creation_tokens": total_cache_creation,
    }

def build_subagent_tokens(usage: Dict[str, Any]) -> Dict[str, int]:
    """Build token metadata dict from PostToolUse(Agent) tool_response.usage."""
    if not usage:
        return {}
    input_t = usage.get("input_tokens", 0)
    cache_read_t = usage.get("cache_read_input_tokens", 0)
    cache_creation_t = usage.get("cache_creation_input_tokens", 0)
    output_t = usage.get("output_tokens", 0)
    prompt_t = input_t + cache_read_t + cache_creation_t
    if not prompt_t and not output_t:
        return {}

    return {
        "prompt_tokens": prompt_t,
        "completion_tokens": output_t,
        "total_tokens": prompt_t + output_t,
        "input_tokens": input_t,
        "cache_read_tokens": cache_read_t,
        "cache_creation_tokens": cache_creation_t,
    }
