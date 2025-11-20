"""
Helper functions for the ADK (Agent Development Kit) metamodel instrumentation.
This module provides utility functions to extract various attributes from agent and tool instances.
"""

from ast import arguments
import json
from typing import Any, Dict, Optional
from monocle_apptrace.instrumentation.metamodel.finish_types import map_adk_finish_reason_to_finish_type
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import set_scope, remove_scope
from monocle_apptrace.instrumentation.common.constants import AGENT_SESSION

def get_model_name(args):
    return args[0].model if hasattr(args[0], 'model') else None

def get_inference_type(arguments):
    """ Find inference type from argument """
    return 'inference.gemini' ## TBD verify non-gemini inference types

def extract_inference_endpoint(instance):
    """ Get inference service end point"""
    if hasattr(instance,'api_client') and hasattr(instance.api_client, '_api_client'):
        if hasattr(instance.api_client._api_client._http_options,'base_url'):
            return instance.api_client._api_client._http_options.base_url
    return None

def extract_message(arguments):
    return str(arguments['args'][0].contents)

def extract_assistant_message(arguments):
    return str(arguments['result'].content.parts)

def update_span_from_llm_response(response, instance):
    meta_dict = {}
    if response is not None and hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        token_usage = response.usage_metadata
        if token_usage is not None:
            meta_dict.update({"completion_tokens": token_usage.candidates_token_count})
            meta_dict.update({"prompt_tokens": token_usage.prompt_token_count })
            meta_dict.update({"total_tokens": token_usage.total_token_count})
    return meta_dict

def extract_finish_reason(arguments):
    if arguments["exception"] is not None:
            return None
    if hasattr(arguments['result'], 'error_code'):
        return arguments['result'].error_code
    return None

def map_finish_reason_to_finish_type(finish_reason:str):
    return map_adk_finish_reason_to_finish_type(finish_reason)

def get_agent_name(instance: Any) -> str:
    """
    Extract the name of the agent from the given instance.

    Args:
        instance: The agent instance to extract name from

    Returns:
        str: The name of the agent, or a default value if not found
    """
    return getattr(instance, 'name', 'unknown_agent')

def get_agent_description(instance: Any) -> str:
    """
    Extract the description of the agent from the given instance.

    Args:
        instance: The agent instance to extract description from

    Returns:
        str: The description of the agent, or a default value if not found
    """
    return getattr(instance, 'description', 'No description available')


def extract_agent_input(arguments: Dict[str, Any]) -> Any:
    """
    Extract the input data from agent arguments.

    Args:
        arguments: Dictionary containing agent call arguments

    Returns:
        Any: The extracted input data
    """
    return [arguments['args'][0].user_content.parts[0].text]

def extract_agent_request_input(arguments: Dict[str, Any]) -> Any:
    """
    Extract the input data from agent request.

    Args:
        arguments: Dictionary containing agent call arguments

    Returns:
        Any: The extracted input data
    """
    return [arguments['kwargs']['new_message'].parts[0].text] if 'new_message' in arguments['kwargs'] else []

def extract_agent_response(result: Any) -> Any:
    """
    Extract the response data from agent result.

    Args:
        result: The result returned by the agent

    Returns:
        Any: The extracted response data
    """
    if result:
        return str(result.content.parts[0].text)
    else:
        return ""

def get_tool_name(instance: Any) -> str:
    """
    Extract the name of the tool from the given instance.

    Args:
        instance: The tool instance to extract name from

    Returns:
        str: The name of the tool, or a default value if not found
    """
    return getattr(instance, 'name', getattr(instance, '__name__', 'unknown_tool'))

def get_tool_description(instance: Any) -> str:
    """
    Extract the description of the tool from the given instance.

    Args:
        instance: The tool instance to extract description from

    Returns:
        str: The description of the tool, or a default value if not found
    """
    return getattr(instance, 'description', getattr(instance, '__doc__', 'No description available'))

def get_source_agent(arguments) -> str:
    """
    Get the name of the source agent (the agent that is calling a tool or delegating to another agent).

    Returns:
        str: The name of the source agent
    """
    return arguments['kwargs']['tool_context'].agent_name

def get_delegating_agent(arguments) -> str:
    """
    Get the name of the delegating agent (the agent that is delegating a task to another agent).

    Args:
        arguments: Dictionary containing agent call arguments 
    Returns:
        str: The name of the delegating agent
    """
    from_agent = arguments['args'][0].agent.name if hasattr(arguments['args'][0], 'agent') else None
    if from_agent is not None:
        if get_agent_name(arguments['instance']) == from_agent:
            return None
    return from_agent

def should_skip_delegation(arguments):
    """
    Determine whether to skip the delegation based on the arguments.

    Args:
        arguments: Dictionary containing agent call arguments

    Returns:
        bool: True if delegation should be skipped, False otherwise
    """
    return get_delegating_agent(arguments) is None

def extract_tool_input(arguments: Dict[str, Any]) -> Any:
    """
    Extract the input data from tool arguments.

    Args:
        arguments: Dictionary containing tool call arguments

    Returns:
        Any: The extracted input data
    """
    return json.dumps(arguments['kwargs'].get('args', {}))

def extract_tool_response(result: Any) -> Any:
    """
    Extract the response data from tool result.

    Args:
        result: The result returned by the tool

    Returns:
        Any: The extracted response data
    """
    return str(result)

def get_target_agent(instance: Any) -> str:
    """
    Extract the name of the target agent (the agent being called/delegated to).

    Args:
        instance: The target agent instance

    Returns:
        str: The name of the target agent
    """
    return getattr(instance, 'name', getattr(instance, '__name__', 'unknown_target_agent'))


class AdkSpanHandler(SpanHandler):
    """Custom span handler for ADK instrumentation that adds session_id scope."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Set session_id scope before tracing begins."""
        session_id_token = None

        if hasattr(instance, '__class__') and instance.__class__.__name__ == 'Runner':
            session_id = kwargs.get('session_id')
            if session_id:
                session_id_token = set_scope(AGENT_SESSION, session_id)

        return session_id_token, None