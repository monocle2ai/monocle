from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import (
    resolve_from_alias,
    get_json_dumps,
)
import logging

logger = logging.getLogger(__name__)

DELEGATION_NAME_PREFIX = "transfer_to_"
ROOT_AGENT_NAME = "AgentsSDK"
AGENTS_AGENT_NAME_KEY = "agent.openai_agents"


def extract_agent_response(response):
    """Extract the final output from an Agents SDK RunResult."""
    try:
        if response is not None and hasattr(response, "final_output"):
            return str(response.final_output)
        elif (
            response is not None
            and hasattr(response, "new_items")
            and response.new_items
        ):
            # Try to extract from new_items if final_output is not available
            last_item = response.new_items[-1]
            if hasattr(last_item, "content"):
                return str(last_item.content)
            elif hasattr(last_item, "text"):
                return str(last_item.text)
        elif (
            response is not None
            and hasattr(response, "next_step")
            and response.next_step
            and hasattr(response.next_step, "output")
            and response.next_step.output
        ):
            return str(response.next_step.output)
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_agent_response: %s", str(e))
    return ""


def extract_agent_input(arguments):
    """Extract the input provided to the agent."""
    try:
        # For Runner.run and Runner.run_sync, the structure is:
        # Runner.run(starting_agent, input, **kwargs)
        # So args[0] = starting_agent, args[1] = input
        if len(arguments["args"]) > 1:
            input_data = arguments["args"][1]
            if isinstance(input_data, str):
                return [input_data]
            elif isinstance(input_data, list):
                # Handle list of input items
                return input_data

        # Fallback to kwargs
        if "original_input" in arguments["kwargs"]:
            input_data = arguments["kwargs"]["original_input"]
            if isinstance(input_data, str):
                return [input_data]
            elif isinstance(input_data, list):
                return input_data
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_agent_input: %s", str(e))
    return []


def get_agent_name(arguments) -> str:
    """Get the name of an agent."""
    instance = None
    # For Runner methods, the agent is passed as an argument, not instance
    if arguments["args"] and len(arguments["args"]) > 0:
        instance = arguments["args"][0]
    else:
        instance = arguments["kwargs"].get("agent")
    if instance is None:
        return "Unknown Agent"
    return get_name(instance)


def get_agent_description(arguments) -> str:
    """Get the description of an agent."""
    instance = None
    # For Runner methods, the agent is passed as an argument, not instance
    if arguments["args"] and len(arguments["args"]) > 0:
        instance = arguments["args"][0]
    else:
        instance = arguments["kwargs"].get("agent")
    if instance is None:
        return ""
    return get_description(instance)


def get_agent_instructions(arguments) -> str:
    """Get the instructions of an agent."""
    instance = None
    # For Runner methods, the agent is passed as an argument, not instance
    if arguments["args"] and len(arguments["args"]) > 0:
        instance = arguments["args"][0]
    else:
        instance = arguments["kwargs"].get("agent")
    if instance is None:
        return ""
    if hasattr(instance, "instructions"):
        instructions = instance.instructions
        if isinstance(instructions, str):
            return instructions
        elif callable(instructions):
            # For dynamic instructions, we can't easily extract them without context
            return "Dynamic instructions (function)"
    return ""


def extract_tool_response(result):
    """Extract response from a tool execution."""
    if result is not None:
        if hasattr(result, "output"):
            return str(result.output)
        elif hasattr(result, "content"):
            return str(result.content)
        else:
            return str(result)
    return None


def extract_tool_input(arguments):
    """Extract input arguments passed to a tool."""
    try:
        # For function tools, the input is typically in args
        if len(arguments["args"]) > 1:
            tool_input = arguments["args"][
                1
            ]  # Second argument is usually the JSON string with params
            if isinstance(tool_input, str):
                return tool_input
            elif isinstance(tool_input, dict):
                return get_json_dumps(tool_input)

        # Fallback to all args
        return str(arguments["args"])
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_input: %s", str(e))
    return []


def get_name(instance):
    """Get the name of an agent or tool instance."""
    if hasattr(instance, "name"):
        return str(instance.name)
    elif hasattr(instance, "__name__"):
        return str(instance.__name__)
    return ""


def get_runner_agent_name(instance) -> str:
    """Get the name of an agent."""
    return get_name(instance)

def get_tool_type(span):
    if (span.attributes.get("is_mcp", False)):
        return "tool.mcp"
    else:
        return "tool.openai_agents"

def get_tool_name(instance) -> str:
    """Get the name of a tool."""
    return get_name(instance)


def is_root_agent_name(instance) -> bool:
    """Check if this is the root agent."""
    return get_name(instance) == ROOT_AGENT_NAME


def get_source_agent() -> str:
    """Get the name of the agent that initiated the request."""
    from_agent = get_value(AGENTS_AGENT_NAME_KEY)
    return from_agent if from_agent is not None else ""


def get_description(instance) -> str:
    """Get the description of an instance."""
    if hasattr(instance, "description"):
        return str(instance.description)
    elif hasattr(instance, "handoff_description"):
        return str(instance.handoff_description)
    elif hasattr(instance, "__doc__") and instance.__doc__:
        return str(instance.__doc__)
    return ""


def get_tool_description(instance) -> str:
    """Get the description of a tool."""
    return get_description(instance)


def extract_handoff_target(arguments):
    """Extract the target agent from a handoff operation."""
    try:
        # Check if this is a handoff by looking at the result
        return arguments.get("result").name
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_handoff_target: %s", str(e))
    return ""


def update_span_from_agent_response(response):
    """Update span with metadata from agent response."""
    meta_dict = {}
    try:
        if response is not None and hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "completion_tokens"):
                meta_dict.update({"completion_tokens": usage.completion_tokens})
            if hasattr(usage, "prompt_tokens"):
                meta_dict.update({"prompt_tokens": usage.prompt_tokens})
            if hasattr(usage, "total_tokens"):
                meta_dict.update({"total_tokens": usage.total_tokens})
    except Exception as e:
        logger.warning(
            "Warning: Error occurred in update_span_from_agent_response: %s", str(e)
        )
    return meta_dict
