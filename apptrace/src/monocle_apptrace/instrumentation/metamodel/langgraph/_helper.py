from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias
import logging
logger = logging.getLogger(__name__)

DELEGATION_NAME_PREFIX = 'transfer_to_'
ROOT_AGENT_NAME = 'LangGraph'
LANGGRAPTH_AGENT_NAME_KEY = "agent.langgraph"

def extract_agent_response(response):
    try:
        if response is not None and 'messages' in response:
            output = response["messages"][-1]
            return str(output.content)
    except Exception as e:
        logger.warning("Warning: Error occurred in handle_response: %s", str(e))
    return ""

def agent_instructions(arguments):
    if callable(arguments['kwargs']['agent'].instructions):
        return arguments['kwargs']['agent'].instructions(arguments['kwargs']['context_variables'])
    else:
        return arguments['kwargs']['agent'].instructions

def extract_agent_input(arguments):
    if arguments['result'] is not None and 'messages' in arguments['result']:
        history = arguments['result']['messages']
        messages = []
        for message in history:
            if hasattr(message, 'content') and hasattr(message, 'type') and message.type == "human":  # Check if the message is a HumanMessage
                messages.append(message.content)
        return messages
    return []

def get_inference_endpoint(arguments):
    inference_endpoint = resolve_from_alias(arguments['instance'].client.__dict__, ['azure_endpoint', 'api_base', '_base_url'])
    return str(inference_endpoint)

def tools(instance):
    if hasattr(instance,'nodes') and ('tools' in instance.nodes):
        tools= instance.nodes['tools']
        if hasattr(tools,'bound') and hasattr(tools.bound,'tools_by_name'):
            return list(tools.bound.tools_by_name.keys())

def update_span_from_llm_response(response):
    meta_dict = {}
    token_usage = None
    if response is not None and "messages" in response:
        token = response["messages"][-1]
        if token.response_metadata is not None:
            token_usage = token.response_metadata["token_usage"]
        if token_usage is not None:
            meta_dict.update({"completion_tokens": token_usage.get('completion_tokens')})
            meta_dict.update({"prompt_tokens": token_usage.get('prompt_tokens')})
            meta_dict.update({"total_tokens": token_usage.get('total_tokens')})
    return meta_dict

def extract_tool_response(result):
    if result is not None and hasattr(result, 'content'):
        return result.content
    return None

def get_status(result):
    if result is not None and hasattr(result, 'status'):
        return result.status
    return None

def extract_tool_input(arguments):
    tool_input = arguments['args'][0]
    if isinstance(tool_input, str):
        return [tool_input]
    else:
        return list(tool_input.values())

def get_name(instance):
    return instance.name if hasattr(instance, 'name') else ""

def get_agent_name(instance) -> str:
    return get_name(instance)

def get_tool_type(span):
    if (span.attributes.get("is_mcp", False)):
        return "tool.mcp"
    else:
        return "tool.langgraph"

def get_tool_name(instance) -> str:
    return get_name(instance)

def is_delegation_tool(instance) -> bool:
    return get_name(instance).startswith(DELEGATION_NAME_PREFIX)

def get_target_agent(instance) -> str:
    return get_name(instance).replace(DELEGATION_NAME_PREFIX, '', 1)

def is_root_agent_name(instance) -> bool:
    return get_name(instance) == ROOT_AGENT_NAME

def get_source_agent() -> str:
    """Get the name of the agent that initiated the request."""
    from_agent = get_value(LANGGRAPTH_AGENT_NAME_KEY)
    return from_agent if from_agent is not None else ""

def get_description(instance) -> str:
    return instance.description if hasattr(instance, 'description') else ""

def get_agent_description(instance) -> str:
    """Get the description of the agent."""
    return get_description(instance)

def get_tool_description(instance) -> str:
    """Get the description of the tool."""
    return get_description(instance)