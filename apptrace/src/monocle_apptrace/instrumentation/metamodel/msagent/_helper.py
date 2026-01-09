"""Helper functions for extracting information from Microsoft Agent Framework objects."""

import logging
from typing import Any, Dict
from opentelemetry.context import get_value

from monocle_apptrace.instrumentation.common.constants import LAST_AGENT_INVOCATION_ID, LAST_AGENT_NAME

logger = logging.getLogger(__name__)

# Context key for accessing agent information (must match processor)
MSAGENT_CONTEXT_KEY = "msagent.agent_info"


def get_agent_from_context() -> Dict[str, Any]:
    """
    Get agent information from OpenTelemetry context.
    This is stored by MSAgentRequestHandler during turn execution.
    
    Returns:
        Dictionary containing agent information (name, instructions, etc.)
    """
    try:
        agent_info = get_value(MSAGENT_CONTEXT_KEY)
        if agent_info:
            return agent_info
        return {}
    except Exception as e:
        logger.warning(f"Error getting agent from context: {e}")
        return {}


def get_agent_name(instance: Any) -> str:
    """Get the name of the agent."""
    try:
        if hasattr(instance, "name"):
            return str(instance.name)
        return "UnknownAgent"
    except Exception as e:
        logger.warning(f"Error getting agent name: {e}")
        return "UnknownAgent"


def get_agent_name_from_context() -> str:
    """
    Get agent name from OpenTelemetry context.
    Used for tool invocations to show which agent owns the tool.
    
    Returns:
        Agent name from context, or empty string if not available
    """
    try:
        agent_info = get_agent_from_context()
        if agent_info and "name" in agent_info:
            return agent_info["name"]
        return ""
    except Exception as e:
        logger.warning(f"Error getting agent name from context: {e}")
        return ""


def get_agent_instructions(instance: Any) -> str:
    """Get the agent's instructions/system prompt."""
    try:
        if hasattr(instance, "instructions"):
            return str(instance.instructions)
        return ""
    except Exception as e:
        logger.warning(f"Error getting agent instructions: {e}")
        return ""


def get_tool_name(instance: Any) -> str:
    """Get the name of the tool."""
    try:
        # For AIFunction, use the name attribute
        if hasattr(instance, "name"):
            return str(instance.name)
        elif hasattr(instance, "__name__"):
            return str(instance.__name__)
        return "UnknownTool"
    except Exception as e:
        logger.warning(f"Error getting tool name: {e}")
        return "UnknownTool"


def get_tool_description(instance: Any) -> str:
    """Get the tool's description."""
    try:
        # For AIFunction, use the description attribute (not __doc__ which is the class docstring)
        if hasattr(instance, "description") and instance.description:
            return str(instance.description)
        # Fallback to the wrapped function's docstring
        elif hasattr(instance, "func") and hasattr(instance.func, "__doc__") and instance.func.__doc__:
            return instance.func.__doc__.strip()
        elif hasattr(instance, "__doc__") and instance.__doc__:
            return instance.__doc__.strip()
        return ""
    except Exception as e:
        logger.warning(f"Error getting tool description: {e}")
        return ""


def extract_agent_input(arguments: Dict[str, Any]) -> str:
    """Extract input from agent invocation arguments."""
    try:
        args = arguments.get("args", ())
        kwargs = arguments.get("kwargs", {})
        
        # Extract input/task from args or kwargs
        if args and len(args) > 0:
            return str(args[0])
        elif "message" in kwargs:
            return str(kwargs["message"])
        elif "input" in kwargs:
            return str(kwargs["input"])
        elif "messages" in kwargs:
            # Handle list of messages
            messages = kwargs["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                # Get the last user message
                return str(messages[-1]) if messages else ""
            return str(messages)
        
        return ""
    except Exception as e:
        logger.warning(f"Error extracting agent input: {e}")
        return ""


def extract_request_agent_input(arguments: Dict[str, Any]) -> str:
    """
    Extract input from agent request (turn level) arguments.
    This is for AGENT_REQUEST spans which represent the complete turn.
    """
    try:
        args = arguments.get("args", ())
        kwargs = arguments.get("kwargs", {})
        
        # For request level, we want the user's query/message
        # which is typically the first argument or 'input'/'message' kwarg
        if args and len(args) > 0:
            input_val = args[0]
            # Handle string inputs
            if isinstance(input_val, str):
                return input_val
            # Handle message objects
            elif hasattr(input_val, "content"):
                return str(input_val.content)
            else:
                return str(input_val)
        
        # Check kwargs for input
        if "input" in kwargs:
            return str(kwargs["input"])
        elif "message" in kwargs:
            return str(kwargs["message"])
        elif "query" in kwargs:
            return str(kwargs["query"])
        elif "messages" in kwargs:
            messages = kwargs["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                # Get the user message (first or last depending on structure)
                first_msg = messages[0] if messages else ""
                if hasattr(first_msg, "content"):
                    return str(first_msg.content)
                return str(first_msg)
            return str(messages)
        
        return ""
    except Exception as e:
        logger.warning(f"Error extracting request agent input: {e}")
        return ""


def extract_agent_response(result: Any, span: Any = None, instance: Any = None, kwargs: Dict[str, Any] = None) -> str:
    """Extract response from agent execution result."""
    try:
        # Handle streaming result (SimpleNamespace from process_msagent_stream)
        if hasattr(result, "type") and hasattr(result, "output_text"):
            if getattr(result, "type", None) == "stream":
                return str(result.output_text) if result.output_text else ""
        
        if result is None:
            return ""
        
        # Check if this is a WorkflowRunResult (list of events)
        # This handles the case when Workflow.run() is instrumented
        if isinstance(result, list) and result:
            # Find the most recent AgentRunEvent with actual content (the final agent response)
            for event in reversed(result):
                event_type = type(event).__name__
                if event_type == "AgentRunEvent" and hasattr(event, "data"):
                    data = event.data
                    if data and isinstance(data, str) and data.strip():
                        return data
            
            # If no AgentRunEvent found, return empty (workflow likely not complete yet)
            return ""
        
        # Handle individual AgentExecutorResponse (from agent invocations)
        # This is what gets passed when individual agents are instrumented
        if hasattr(result, "agent_run_response"):
            agent_run_response = result.agent_run_response
            
            # First, try to extract from agent_run_response.content
            if hasattr(agent_run_response, "content"):
                content = agent_run_response.content
                if isinstance(content, str) and content.strip():
                    return content
                elif hasattr(content, "text") and content.text:
                    return str(content.text)
            
            # If content is None or empty, check messages in agent_run_response
            if hasattr(agent_run_response, "messages") and agent_run_response.messages:
                for msg in reversed(agent_run_response.messages):
                    if hasattr(msg, "role") and msg.role == "assistant":
                        if hasattr(msg, "content") and msg.content:
                            if isinstance(msg.content, str):
                                return msg.content
                            elif isinstance(msg.content, list):
                                texts = []
                                for part in msg.content:
                                    if hasattr(part, "text"):
                                        texts.append(str(part.text))
                                    elif isinstance(part, dict) and "text" in part:
                                        texts.append(str(part["text"]))
                                if texts:
                                    return " ".join(texts)
                        break
            
            # If still nothing, check full_conversation on AgentExecutorResponse
            # This contains all messages including those before the current response
            if hasattr(result, "full_conversation") and result.full_conversation:
                # Look for the last assistant message with actual text content
                for msg in reversed(result.full_conversation):
                    if hasattr(msg, "role") and msg.role == "assistant":
                        # Check if message has content
                        if hasattr(msg, "content") and msg.content:
                            if isinstance(msg.content, str) and msg.content.strip():
                                return msg.content
                            elif isinstance(msg.content, list):
                                texts = []
                                for part in msg.content:
                                    if hasattr(part, "text") and part.text:
                                        texts.append(str(part.text))
                                    elif isinstance(part, dict) and "text" in part and part["text"]:
                                        texts.append(str(part["text"]))
                                if texts:
                                    return " ".join(texts)
            
            return ""
        
        # Check for accumulated_text attribute (set by wrapper for streaming)
        if hasattr(result, "accumulated_text"):
            return str(result.accumulated_text) if result.accumulated_text else ""
        
        # Check for direct AgentRunResponse (non-streaming) - has 'content' attribute
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, str) and content.strip():
                return content
            elif hasattr(content, "text"):
                return str(content.text)
            # If content is None but we have messages, get last assistant message
            elif content is None and hasattr(result, "messages") and result.messages:
                for msg in reversed(result.messages):
                    if hasattr(msg, "role") and msg.role == "assistant":
                        if hasattr(msg, "content") and msg.content:
                            if isinstance(msg.content, str):
                                return msg.content
                            elif isinstance(msg.content, list):
                                texts = []
                                for part in msg.content:
                                    if hasattr(part, "text"):
                                        texts.append(str(part.text))
                                    elif isinstance(part, dict) and "text" in part:
                                        texts.append(str(part["text"]))
                                if texts:
                                    return " ".join(texts)
                        break
                return ""
            else:
                return str(content) if content else ""
        
        # Check for AgentRunResponseUpdate (streaming) - has 'text' attribute
        if hasattr(result, "text"):
            text = result.text
            return str(text) if text else ""
        
        # Check if it's a simple string
        if isinstance(result, str):
            return result
        
        # For unhandled types, return empty string to avoid showing internal details
        return ""
        
    except Exception as e:
        logger.warning(f"Error extracting agent response: {e}")
        return ""


def extract_from_agent_invocation_id(parent_span):
    if parent_span is not None:
        return parent_span.attributes.get(LAST_AGENT_INVOCATION_ID)
    return None

def extract_from_agent_name(parent_span):
    if parent_span is not None:
        return parent_span.attributes.get(LAST_AGENT_NAME)
    return None

def extract_tool_input(arguments: Dict[str, Any]) -> str:
    """Extract input from tool invocation arguments."""
    try:
        kwargs = arguments.get("kwargs", {})
        args = arguments.get("args", ())
        
        # For AIFunction.invoke, extract the 'arguments' parameter which contains the actual tool inputs
        if kwargs and "arguments" in kwargs:
            arg_value = kwargs["arguments"]
            # If it's a Pydantic model, convert to dict
            if hasattr(arg_value, "model_dump"):
                return str(arg_value.model_dump())
            elif hasattr(arg_value, "dict"):
                return str(arg_value.dict())
            else:
                return str(arg_value)
        # Fallback to regular kwargs/args
        elif kwargs:
            # Filter out internal parameters like tool_call_id
            filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_") and k != "tool_call_id"}
            return str(filtered_kwargs) if filtered_kwargs else str(kwargs)
        elif args:
            return str(args)
        
        return ""
    except Exception as e:
        logger.warning(f"Error extracting tool input: {e}")
        return ""


def extract_tool_response(result: Any, span: Any = None) -> str:
    """Extract response from tool execution result."""
    try:
        if result is None:
            return ""
            
        # Check if it's a coroutine (shouldn't be, but handle it)
        if hasattr(result, "__name__") and "coroutine" in str(type(result)):
            return ""
            
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return str(result)
        elif isinstance(result, (list, tuple)):
            return str(result)
        else:
            # Avoid object representations
            result_str = str(result)
            if result_str and not result_str.startswith("<") and "object at 0x" not in result_str and "coroutine" not in result_str:
                return result_str
            return ""
    except Exception as e:
        logger.warning(f"Error extracting tool response: {e}")
        return ""


# Chat Client specific helpers (for OpenAIBaseChatClient.get_response/get_streaming_response)

def get_chat_client_name(instance: Any) -> str:
    """
    Get the name/identifier of the chat client.
    First checks context for agent name (follows ADK pattern where invocation has agent name).
    Falls back to chat client identifier if no agent context.
    """
    try:
        # First check if agent name is available in context
        agent_info = get_agent_from_context()
        if agent_info and "name" in agent_info:
            return agent_info["name"]
        
        # Fallback to chat client identifier
        if hasattr(instance, "service_url"):
            # service_url might be a property/method, call it if callable
            url = instance.service_url
            if callable(url):
                url = url()
            return f"ChatClient({url})"
        elif hasattr(instance, "__class__"):
            return instance.__class__.__name__
        return "ChatClient"
    except Exception as e:
        logger.warning(f"Error getting chat client name: {e}")
        return "ChatClient"


def get_chat_client_model(instance: Any) -> str:
    """Get the model identifier from the chat client."""
    try:
        # Try to get default model settings
        if hasattr(instance, "_default_chat_options") and instance._default_chat_options:
            if hasattr(instance._default_chat_options, "model_id"):
                return str(instance._default_chat_options.model_id)
        # Fallback
        return "unknown_model"
    except Exception as e:
        logger.warning(f"Error getting chat client model: {e}")
        return "unknown_model"
def get_from_agent_name(arguments: Dict[str, Any]) -> str:
    """Extract delegating agent name from parent span attributes."""
    try:
        parent_span = arguments.get('parent_span')
        if parent_span:
            return parent_span.attributes.get('monocle.last.agent.name', '')
        return ''
    except Exception as e:
        logger.warning(f"Error extracting from_agent name: {e}")
        return ''
    
def get_from_agent_span_id(arguments: Dict[str, Any]) -> str:
    """Extract delegating agent invocation ID from parent span attributes."""
    try:
        parent_span = arguments.get('parent_span')
        if parent_span:
            return parent_span.attributes.get('monocle.last.agent.invocation.id', '')
        return ''
    except Exception as e:
        logger.warning(f"Error extracting from_agent invocation id: {e}")
        return ''
    
def extract_chat_client_input(arguments: Dict[str, Any]) -> str:
    """Extract input from chat client get_response/get_streaming_response arguments."""
    try:
        args = arguments.get("args", ())
        kwargs = arguments.get("kwargs", {})
        
        # Chat client methods receive 'messages' parameter
        messages = None
        if "messages" in kwargs:
            messages = kwargs["messages"]
        elif args and len(args) > 0:
            # First arg is messages
            messages = args[0]
        
        if messages:
            if isinstance(messages, list) and len(messages) > 0:
                # Get the last user message (most recent input)
                last_msg = messages[-1]
                
                # Try to extract content from the message object
                if hasattr(last_msg, "content"):
                    content = last_msg.content
                    # Content might be a string or a list of content parts
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Extract text from content parts
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and "text" in part:
                                text_parts.append(part["text"])
                            elif hasattr(part, "text"):
                                text_parts.append(str(part.text))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        return " ".join(text_parts) if text_parts else str(content)
                    else:
                        return str(content)
                
                # If no content attribute, try text
                if hasattr(last_msg, "text"):
                    return str(last_msg.text)
                
                # Last resort - string conversion but avoid object repr
                msg_str = str(last_msg)
                if not msg_str.startswith("<") and not "object at 0x" in msg_str:
                    return msg_str
                
            return str(messages) if not str(messages).startswith("<") else ""
        
        return ""
    except Exception as e:
        logger.warning(f"Error extracting chat client input: {e}")
        return ""


def extract_chat_client_response(result: Any, span: Any = None) -> str:
    """Extract response from chat client result (ChatResponseUpdate)."""
    try:
        if result is None:
            return ""
        
        # Handle streaming result (SimpleNamespace from process_msagent_stream)
        if hasattr(result, "type") and hasattr(result, "output_text"):
            if getattr(result, "type", None) == "stream":
                return str(result.output_text) if result.output_text else ""
        
        # Check for accumulated_text attribute (set by wrapper for streaming)
        if hasattr(result, "accumulated_text"):
            return str(result.accumulated_text) if result.accumulated_text else ""
        
        # Handle ChatResponseUpdate (streaming) - has 'text' attribute
        if hasattr(result, "text"):
            text = result.text
            return str(text) if text else ""
        
        # Check if it's a simple string
        if isinstance(result, str):
            return result
        
        # Fallback - but avoid object representations
        result_str = str(result)
        if result_str and not result_str.startswith("<") and "object at 0x" not in result_str:
            return result_str
            
        return ""
    except Exception as e:
        logger.warning(f"Error extracting chat client response: {e}")
        return ""
