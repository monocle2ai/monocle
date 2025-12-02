from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias
import logging
logger = logging.getLogger(__name__)

DELEGATION_NAME_PREFIX = 'delegate_to_'
ROOT_CREW_NAME = 'CrewAI'
CREW_AI_AGENT_NAME_KEY = "agent.crewai"

def extract_agent_response(response):
    try:
        if response is not None:
            if hasattr(response, 'raw'):
                return str(response.raw)
            elif hasattr(response, 'output'):
                return str(response.output)
            elif isinstance(response, str):
                return response
    except Exception as e:
        logger.warning("Warning: Error occurred in handle_response: %s", str(e))
    return ""

def agent_instructions(arguments):
    try:
        if 'kwargs' in arguments and arguments['kwargs'] is not None:
            if 'agent' in arguments['kwargs']:
                agent = arguments['kwargs']['agent']
                if hasattr(agent, 'backstory'):
                    return agent.backstory
                elif hasattr(agent, 'role'):
                    return agent.role
        return ""
    except Exception as e:
        logger.warning("Error extracting agent instructions: %s", str(e))
        return ""

def extract_request_agent_input(arguments):
    try:
        if arguments['kwargs'] is not None and 'inputs' in arguments['kwargs']:
            inputs = arguments['kwargs']['inputs']
            if isinstance(inputs, dict):
                return list(inputs.values())
            elif isinstance(inputs, list):
                return inputs
            else:
                return [str(inputs)]
        return []
    except Exception as e:
        logger.warning("Error extracting request agent input: %s", str(e))
        return []

def extract_agent_input(arguments):
    try:
        # Check kwargs first (CrewAI passes task in kwargs)
        if arguments['kwargs'] is not None:
            kwargs = arguments['kwargs']
            
            # Extract task description if available
            if 'task' in kwargs and hasattr(kwargs['task'], 'description'):
                return [kwargs['task'].description]
            
            # Fallback to any other string values in kwargs
            input_values = []
            for key, value in kwargs.items():
                if isinstance(value, str) and value.strip():
                    input_values.append(f"{key}: {value}")
                elif hasattr(value, 'description') and value.description:
                    input_values.append(value.description)
            
            if input_values:
                return input_values
        
        # Fallback to original args logic
        if arguments['args'] is not None and len(arguments['args']) > 0:
            if isinstance(arguments['args'][0], dict):
                return list(arguments['args'][0].values())
            else:
                return [str(arg) for arg in arguments['args']]
        
        return []
    except Exception as e:
        logger.warning("Error extracting agent input: %s", str(e))
        return []

def get_inference_endpoint(arguments):
    try:
        if 'instance' in arguments and hasattr(arguments['instance'], 'llm'):
            llm = arguments['instance'].llm
            inference_endpoint = resolve_from_alias(llm.__dict__, ['azure_endpoint', 'api_base', '_base_url'])
            return str(inference_endpoint)
        return ""
    except Exception as e:
        logger.warning("Error getting inference endpoint: %s", str(e))
        return ""

def tools(instance):
    try:
        if hasattr(instance, 'tools') and instance.tools:
            if isinstance(instance.tools, list):
                return [tool.name if hasattr(tool, 'name') else str(tool) for tool in instance.tools]
        return []
    except Exception as e:
        logger.warning("Error extracting tools: %s", str(e))
        return []

def update_span_from_llm_response(response):
    meta_dict = {}
    try:
        if response is not None and hasattr(response, 'usage'):
            usage = response.usage
            if hasattr(usage, 'completion_tokens'):
                meta_dict.update({"completion_tokens": usage.completion_tokens})
            if hasattr(usage, 'prompt_tokens'):
                meta_dict.update({"prompt_tokens": usage.prompt_tokens})
            if hasattr(usage, 'total_tokens'):
                meta_dict.update({"total_tokens": usage.total_tokens})
    except Exception as e:
        logger.warning("Error updating span from LLM response: %s", str(e))
    return meta_dict

def extract_tool_response(result):
    try:
        if result is not None:
            if hasattr(result, 'output'):
                return str(result.output)
            elif hasattr(result, 'content'):
                return str(result.content)
            elif isinstance(result, str):
                return result
            else:
                return str(result)
    except Exception as e:
        logger.warning("Error extracting tool response: %s", str(e))
    return ""

def get_status(result):
    try:
        if result is not None and hasattr(result, 'status'):
            return result.status
        return None
    except Exception as e:
        logger.warning("Error getting status: %s", str(e))
        return None

def extract_tool_input(arguments):
    try:
        # Handle different tool input patterns
        if arguments['args'] and len(arguments['args']) > 0:
            tool_input = arguments['args'][0]
            # If it's a dict, return formatted key-value pairs
            if isinstance(tool_input, dict):
                return [f"{k}: {v}" for k, v in tool_input.items()]
            else:
                return [str(tool_input)]
        elif arguments['kwargs']:
            tool_input = arguments['kwargs'].copy()
            # Remove CrewAI-specific parameters that aren't part of tool input
            for param in ['context', 'agent', 'run_manager', 'config', 'callbacks']:
                tool_input.pop(param, None)
            if tool_input:
                return [f"{k}: {v}" for k, v in tool_input.items()]
        return []
    except Exception as e:
        logger.warning("Error extracting tool input: %s", str(e))
        return []

def get_name(instance):
    try:
        # For CrewAI agents, prioritize role as the name
        if hasattr(instance, 'role') and instance.role:
            return instance.role
        # For tools, prioritize tool-specific naming patterns
        elif hasattr(instance, 'name') and instance.name:
            return instance.name
        elif hasattr(instance, 'func') and hasattr(instance.func, '__name__'):
            # For function-based tools
            return instance.func.__name__
        elif hasattr(instance, '__class__') and hasattr(instance.__class__, '__name__'):
            # For class-based tools, use class name as fallback
            class_name = instance.__class__.__name__
            if class_name not in ['Tool', 'BaseTool', 'StructuredTool', 'Agent']:
                return class_name
        elif hasattr(instance, 'description'):
            # Extract first meaningful part of description as fallback
            desc = instance.description.strip()
            if desc:
                # Take first sentence or first 50 chars, whichever is shorter
                first_sentence = desc.split('.')[0]
                return first_sentence[:50] if len(first_sentence) > 50 else first_sentence
        else:
            return "Unknown"
    except Exception as e:
        logger.warning("Error getting name: %s", str(e))
        return "Unknown"

def get_agent_name(instance) -> str:
    return get_name(instance)

def get_tool_type(span):
    try:
        if (span.attributes.get("is_mcp", False)):
            return "tool.mcp"
        # Check if it's a LangChain tool being used by CrewAI
        elif span.attributes.get("langchain_tool", False):
            return "tool.langchain"
        else:
            return "tool.crewai"
    except Exception as e:
        logger.warning("Error getting tool type: %s", str(e))
        return "tool.crewai"

def get_tool_name(instance) -> str:
    return get_name(instance)

def is_delegation_task(instance) -> bool:
    try:
        return get_name(instance).startswith(DELEGATION_NAME_PREFIX)
    except Exception:
        return False

def get_target_agent(instance) -> str:
    try:
        return get_name(instance).replace(DELEGATION_NAME_PREFIX, '', 1)
    except Exception:
        return ""

def is_root_crew_name(instance) -> bool:
    try:
        return get_name(instance) == ROOT_CREW_NAME
    except Exception:
        return False

def get_source_agent() -> str:
    """Get the name of the agent that initiated the request."""
    try:
        from_agent = get_value(CREW_AI_AGENT_NAME_KEY)
        return from_agent if from_agent is not None else ""
    except Exception as e:
        logger.warning("Error getting source agent: %s", str(e))
        return ""

def get_description(instance) -> str:
    try:
        if hasattr(instance, 'description'):
            return instance.description
        elif hasattr(instance, 'backstory'):
            return instance.backstory
        else:
            return ""
    except Exception as e:
        logger.warning("Error getting description: %s", str(e))
        return ""

def get_agent_description(instance) -> str:
    """Get the description of the agent."""
    return get_description(instance)

def get_tool_description(instance) -> str:
    """Get the description of the tool."""
    return get_description(instance)