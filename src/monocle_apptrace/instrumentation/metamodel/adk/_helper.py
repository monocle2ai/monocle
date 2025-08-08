"""
Helper functions for the ADK (Agent Development Kit) metamodel instrumentation.
This module provides utility functions to extract various attributes from agent and tool instances.
"""

from ast import arguments
from typing import Any, Dict, Optional


def get_agent_name(instance: Any) -> str:
    """
    Extract the name of the agent from the given instance.
    
    Args:
        instance: The agent instance to extract name from
        
    Returns:
        str: The name of the agent, or a default value if not found
    """
    # TODO: Implement logic to extract agent name from instance
    return getattr(instance, 'name', 'unknown_agent')


def get_agent_description(instance: Any) -> str:
    """
    Extract the description of the agent from the given instance.
    
    Args:
        instance: The agent instance to extract description from
        
    Returns:
        str: The description of the agent, or a default value if not found
    """
    # TODO: Implement logic to extract agent description from instance
    return getattr(instance, 'description', 'No description available')


def extract_agent_input(arguments: Dict[str, Any]) -> Any:
    """
    Extract the input data from agent arguments.
    
    Args:
        arguments: Dictionary containing agent call arguments
        
    Returns:
        Any: The extracted input data
    """
    return arguments['args'][0].user_content.parts[0].text


def extract_agent_response(result: Any) -> Any:
    """
    Extract the response data from agent result.
    
    Args:
        result: The result returned by the agent
        
    Returns:
        Any: The extracted response data
    """
    # TODO: Implement logic to extract agent response from result
    if result:
        return result.content.parts[0].text


def get_tool_name(instance: Any) -> str:
    """
    Extract the name of the tool from the given instance.
    
    Args:
        instance: The tool instance to extract name from
        
    Returns:
        str: The name of the tool, or a default value if not found
    """
    # TODO: Implement logic to extract tool name from instance
    return getattr(instance, 'name', getattr(instance, '__name__', 'unknown_tool'))


def get_tool_description(instance: Any) -> str:
    """
    Extract the description of the tool from the given instance.
    
    Args:
        instance: The tool instance to extract description from
        
    Returns:
        str: The description of the tool, or a default value if not found
    """
    # TODO: Implement logic to extract tool description from instance
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

def extract_tool_input(arguments: Dict[str, Any]) -> Any:
    """
    Extract the input data from tool arguments.
    
    Args:
        arguments: Dictionary containing tool call arguments
        
    Returns:
        Any: The extracted input data
    """
    # TODO: Implement logic to extract tool input from arguments
    # This might involve extracting function arguments, parameters, etc.
    return str(arguments['kwargs'].get('args'))

def extract_tool_response(result: Any) -> Any:
    """
    Extract the response data from tool result.
    
    Args:
        result: The result returned by the tool
        
    Returns:
        Any: The extracted response data
    """
    # TODO: Implement logic to extract tool response from result
    #if isinstance(result, dict):
    #    return result
    return str(result)


def get_target_agent(instance: Any) -> str:
    """
    Extract the name of the target agent (the agent being called/delegated to).
    
    Args:
        instance: The target agent instance
        
    Returns:
        str: The name of the target agent
    """
    # TODO: Implement logic to extract target agent name from instance
    return getattr(instance, 'name', getattr(instance, '__name__', 'unknown_target_agent'))
