"""Helper functions for extracting information from Microsoft Agent Framework objects."""

import logging
from typing import Any, Dict
from urllib.parse import urlparse
from opentelemetry.context import get_value

from monocle_apptrace.instrumentation.common.constants import (
    AGENT_PREFIX_KEY,
    AGENT_SESSION,
    INFERENCE_AGENT_DELEGATION,
    INFERENCE_TOOL_CALL,
    INFERENCE_TURN_END,
)
from monocle_apptrace.instrumentation.common.utils import (
    get_exception_message,
    get_json_dumps,
    get_status_code,
    set_scope,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import (
    map_msagent_finish_reason_to_finish_type
)

logger = logging.getLogger(__name__)

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


def get_tool_owner_agent_name(arguments: Dict[str, Any]) -> str:
    """Resolve tool-owner agent name with context-first, parent-span fallback strategy."""
    try:
        agent_name = get_agent_name_from_context()
        if agent_name:
            return agent_name

        parent_span = arguments.get("parent_span")
        if parent_span:
            # Preferred propagated key.
            propagated_name = parent_span.attributes.get("monocle.last.agent.name", "")
            if propagated_name:
                return propagated_name

            # Fallback when tool span is directly under an invocation span.
            direct_parent_name = parent_span.attributes.get("entity.1.name", "")
            if direct_parent_name:
                return direct_parent_name
        return ""
    except Exception as e:
        logger.warning(f"Error resolving tool owner agent name: {e}")
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


def extract_agent_response(arguments: Any) -> str:
    """Extract response from agent execution result."""
    try:
        result = arguments["result"]
        # Get service_thread_id when Azure API returns it
        thread = arguments["kwargs"].get("thread")
        if thread is not None and hasattr(thread, "service_thread_id"):
            service_thread_id = thread.service_thread_id
            # Set session scope with the real Azure thread ID when it becomes available
            # This adds it to baggage for subsequent spans AND directly to current span
            if service_thread_id:
                # Only set scope if not already set
                existing_scope = get_value(AGENT_SESSION)
                if not existing_scope:
                    set_scope(AGENT_SESSION, service_thread_id)
                    # Also set directly on current span if available
                    span = arguments.get("span")
                    if span is not None:
                        span.set_attribute(f"scope.{AGENT_SESSION}", service_thread_id)
        
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


def _first_non_empty_attr(target: Any, attr_names: list[str]) -> Any:
    """Return the first non-empty attribute value from the provided names."""
    if target is None:
        return None
    for attr_name in attr_names:
        attr_value = getattr(target, attr_name, None)
        if attr_value:
            return attr_value
    return None


def get_chat_client_model(instance: Any) -> str:
    """Get the model identifier from the chat client."""
    try:
        candidate_attrs = ["model_id", "model", "deployment_name", "azure_deployment", "deployment"]

        # Try to get default model settings
        if hasattr(instance, "_default_chat_options") and instance._default_chat_options:
            model_value = _first_non_empty_attr(instance._default_chat_options, candidate_attrs)
            if model_value:
                return str(model_value)

        model_value = _first_non_empty_attr(instance, candidate_attrs)
        if model_value:
            return str(model_value)

        nested_client = getattr(instance, "client", None) or getattr(instance, "_client", None)
        model_value = _first_non_empty_attr(nested_client, candidate_attrs)
        if model_value:
            return str(model_value)
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


def _extract_text_from_part(part: Any) -> str:
    """Extract text from a content part, including one nested text/value layer."""
    if part is None:
        return ""

    if isinstance(part, str):
        return part

    if isinstance(part, dict):
        if isinstance(part.get("text"), str):
            return part.get("text", "")
        text_obj = part.get("text")
        if isinstance(text_obj, dict) and isinstance(text_obj.get("value"), str):
            return text_obj.get("value", "")
        if isinstance(part.get("value"), str):
            return part.get("value", "")
        if isinstance(part.get("content"), str):
            return part.get("content", "")

    text_attr = getattr(part, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    text_value = getattr(text_attr, "value", None)
    if isinstance(text_value, str):
        return text_value

    value_attr = getattr(part, "value", None)
    if isinstance(value_attr, str):
        return value_attr

    content_attr = getattr(part, "content", None)
    if isinstance(content_attr, str):
        return content_attr

    return ""


def _extract_text_from_content(content: Any) -> str:
    """Extract text from content that may be str/list/object with one nested layer."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for part in content:
            part_text = _extract_text_from_part(part)
            if part_text:
                texts.append(part_text)
        return " ".join(texts)

    if isinstance(content, dict):
        if isinstance(content.get("content"), list):
            return _extract_text_from_content(content.get("content"))
        return _extract_text_from_part(content)

    return _extract_text_from_part(content)


def _extract_text_from_output_items(output_items: Any) -> str:
    """Extract assistant text from output items using one-level nested traversal."""
    if not isinstance(output_items, list) or len(output_items) == 0:
        return ""

    for item in reversed(output_items):
        # Dict form
        if isinstance(item, dict):
            item_type = item.get("type")
            role = item.get("role")
            if item_type == "message" or role == "assistant":
                text = _extract_text_from_content(item.get("content"))
                if text:
                    return text
            # Fallback for non-message items carrying output text
            text = _extract_text_from_content(item.get("output"))
            if text:
                return text

        # Object form
        item_type = getattr(item, "type", None)
        role = getattr(item, "role", None)
        if item_type == "message" or role == "assistant":
            text = _extract_text_from_content(getattr(item, "content", None))
            if text:
                return text
        text = _extract_text_from_content(getattr(item, "output", None))
        if text:
            return text

    return ""
    
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

                # Dict-based message structure
                if isinstance(last_msg, dict):
                    text = _extract_text_from_content(last_msg.get("content"))
                    if text:
                        return text
                    if isinstance(last_msg.get("text"), str):
                        return last_msg.get("text", "")
                
                # Try to extract content from the message object
                if hasattr(last_msg, "content"):
                    content = last_msg.content
                    # Content might be a string or a list of content parts
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        deep_text = _extract_text_from_content(content)
                        return deep_text if deep_text else str(content)
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
                if result.output_text:
                    return str(result.output_text)

                # Go one level deeper for stream responses that keep content in response/output.
                response_obj = getattr(result, "response", None)
                if response_obj is not None:
                    nested_output = getattr(response_obj, "output", None)
                    nested_text = _extract_text_from_output_items(nested_output)
                    if nested_text:
                        return nested_text
        
        # Check for accumulated_text attribute (set by wrapper for streaming)
        if hasattr(result, "accumulated_text"):
            return str(result.accumulated_text) if result.accumulated_text else ""

        # Some streaming adapters expose response.output on the result directly.
        if hasattr(result, "response"):
            nested_output = getattr(result.response, "output", None)
            nested_text = _extract_text_from_output_items(nested_output)
            if nested_text:
                return nested_text
        if hasattr(result, "output"):
            nested_text = _extract_text_from_output_items(result.output)
            if nested_text:
                return nested_text
        
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


def uses_chat_client(instance: Any) -> bool:
    """Check if ChatAgent instance uses AzureOpenAIChatClient.
    
    Returns True if the agent uses AzureOpenAIChatClient,
    False if it uses AzureOpenAIAssistantsClient or unknown.
    """
    try:
        if hasattr(instance, "_client"):
            client_class = instance._client.__class__.__name__
            return client_class == "AzureOpenAIChatClient"
        return False
    except Exception as e:
        logger.debug(f"Error detecting client type: {e}")
        return False


# Additional helper functions for INFERENCE entity

def get_inference_type(instance):
    """Get inference type (azure_openai, openai, etc.) from instance."""
    try:
        # For MS Agent, check if it's Azure OpenAI
        if hasattr(instance, '_client') and hasattr(instance._client, '_api_version'):
            return 'azure_openai'
        # Check OTEL_PROVIDER_NAME attribute
        if hasattr(instance, 'OTEL_PROVIDER_NAME'):
            provider = instance.OTEL_PROVIDER_NAME
            if provider == 'openai':
                return 'azure_openai'  # MS Agent typically uses Azure OpenAI
            return provider
        # Default to azure_openai for MS Agent since it primarily uses Azure
        return 'azure_openai'
    except Exception as e:
        logger.warning(f"Error getting inference type: {e}")
        return 'azure_openai'


def extract_provider_name(instance):
    """Extract provider name from instance."""
    try:
        # First try client._base_url.host (for AzureOpenAIAssistantsClient)
        if hasattr(instance, 'client') and hasattr(instance.client, '_base_url'):
            base_url = instance.client._base_url
            if hasattr(base_url, 'host'):
                return base_url.host
        
        # Try OTEL_PROVIDER_NAME attribute as fallback
        if hasattr(instance, 'OTEL_PROVIDER_NAME'):
            return instance.OTEL_PROVIDER_NAME
        
        # Try client.base_url (for AzureOpenAIAssistantsClient)
        if hasattr(instance, 'client') and hasattr(instance.client, 'base_url'):
            base_url = instance.client.base_url
            if hasattr(base_url, 'host'):
                return base_url.host
            elif isinstance(base_url, str):
                parsed = urlparse(base_url)
                if parsed.hostname:
                    return parsed.hostname
        
        # Try base_url attribute directly
        if hasattr(instance, 'base_url'):
            base_url = instance.base_url
            if base_url and base_url != 'None':
                if hasattr(base_url, 'host'):
                    return base_url.host
                elif isinstance(base_url, str):
                    parsed = urlparse(base_url)
                    if parsed.hostname:
                        return parsed.hostname
        
        # Try _client.base_url
        if hasattr(instance, '_client') and hasattr(instance._client, 'base_url'):
            base_url = instance._client.base_url
            if hasattr(base_url, 'host'):
                return base_url.host
            elif isinstance(base_url, str):
                parsed = urlparse(base_url)
                if parsed.hostname:
                    return parsed.hostname
        return None
    except Exception as e:
        logger.warning(f"Error extracting provider name: {e}")
        return None


def extract_inference_endpoint(instance):
    """Extract inference endpoint from instance."""
    try:
        # Try client.base_url (for AzureOpenAIAssistantsClient)
        if hasattr(instance, 'client') and hasattr(instance.client, 'base_url'):
            base_url = instance.client.base_url
            if base_url:
                return str(base_url)
        
        # Try base_url attribute directly
        if hasattr(instance, 'base_url'):
            base_url = instance.base_url
            if base_url and base_url != 'None':
                return str(base_url)
        
        # Try _client.base_url
        if hasattr(instance, '_client') and hasattr(instance._client, 'base_url'):
            base_url = str(instance._client.base_url)
            return base_url
        
        return extract_provider_name(instance)
    except Exception as e:
        logger.warning(f"Error extracting inference endpoint: {e}")
        return None


def extract_messages(kwargs):
    """Extract messages from kwargs for input."""
    try:
        messages = []
        
        # Check for options dictionary with instructions
        if 'options' in kwargs:
            options = kwargs['options']
            if isinstance(options, dict) and 'instructions' in options:
                messages.append({'system': options['instructions']})
        elif 'instructions' in kwargs:
            messages.append({'system': kwargs.get('instructions', {})})
        
        # Check for messages list
        if 'messages' in kwargs:
            msgs = kwargs['messages']
            if isinstance(msgs, list):
                for msg in msgs:
                    # Handle message objects with text attribute
                    if hasattr(msg, 'text'):
                        role = getattr(msg, 'role', 'user')
                        messages.append({role: msg.text})
                    elif isinstance(msg, dict):
                        messages.append(msg)
                    else:
                        messages.append({'user': str(msg)})
            else:
                return get_json_dumps(msgs)
        
        # Fallback to input
        if not messages and 'input' in kwargs:
            if isinstance(kwargs['input'], str):
                messages.append({'user': kwargs.get('input', "")})
            elif isinstance(kwargs['input'], list):
                messages.extend(kwargs['input'])
        
        return get_json_dumps(messages) if messages else ""
    except Exception as e:
        logger.warning(f"Error extracting messages: {e}")
        return ""


def map_finish_reason_to_finish_type(finish_reason):
    """Map finish_reason to finish_type using MS Agent Framework mapping."""
    return map_msagent_finish_reason_to_finish_type(finish_reason)


def extract_model_name(instance, kwargs):
    """Extract model name from instance or kwargs."""
    try:
        # First try model_id attribute from instance
        if hasattr(instance, 'model_id'):
            return instance.model_id
        
        # Try kwargs
        if kwargs:
            from monocle_apptrace.instrumentation.common.utils import resolve_from_alias
            model_name = resolve_from_alias(
                kwargs,
                ["model", "model_name", "model_id", "endpoint_name", "deployment_name"],
            )
            if model_name:
                return model_name
        
        return None
    except Exception as e:
        logger.warning(f"Error extracting model name: {e}")
        return None


def extract_model_type(instance, kwargs):
    """Extract model type from instance or kwargs."""
    try:
        model_name = extract_model_name(instance, kwargs)
        if model_name:
            return f"model.llm.{model_name}"
        return None
    except Exception as e:
        logger.warning(f"Error extracting model type: {e}")
        return None


def _get_field(value, key, default=None):
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _response_contains_tool_calls(response):
    try:
        if response is None:
            return False

        if _as_list(_get_field(response, "tools")):
            return True

        for output_item in _as_list(_get_field(response, "output")):
            item_type = _get_field(output_item, "type")
            if item_type in ("function_call", "tool_call", "tool_calls"):
                return True
            if _as_list(_get_field(output_item, "tool_calls")):
                return True
            if _get_field(output_item, "call_id") and _get_field(output_item, "name"):
                return True

        required_action = _get_field(response, "required_action")
        submit_tool_outputs = _get_field(required_action, "submit_tool_outputs")
        if _as_list(_get_field(submit_tool_outputs, "tool_calls")):
            return True

        for message in _as_list(_get_field(response, "messages")):
            if _as_list(_get_field(message, "tool_calls")):
                return True
            for content in _as_list(_get_field(message, "contents")):
                content_type = _get_field(content, "type") or type(content).__name__
                if content_type in ("FunctionCallContent", "function_call", "tool_call", "tool_calls"):
                    return True
                if _get_field(content, "call_id") and _get_field(content, "name"):
                    return True

        for choice in _as_list(_get_field(response, "choices")):
            finish_reason = _get_field(choice, "finish_reason")
            if finish_reason in ("tool_calls", "function_call"):
                return True
            if _as_list(_get_field(_get_field(choice, "message"), "tool_calls")):
                return True
            if _as_list(_get_field(_get_field(choice, "delta"), "tool_calls")):
                return True
    except Exception as exc:
        logger.debug(f"Error while checking tool call response: {exc}")
    return False


def _extract_first_tool_name(response):
    try:
        for tool in _as_list(_get_field(response, "tools")):
            tool_name = _get_field(tool, "name")
            if tool_name:
                return str(tool_name)
        for output_item in _as_list(_get_field(response, "output")):
            tool_name = _get_field(output_item, "name")
            if tool_name:
                return str(tool_name)
        required_action = _get_field(response, "required_action")
        submit_tool_outputs = _get_field(required_action, "submit_tool_outputs")
        for tool_call in _as_list(_get_field(submit_tool_outputs, "tool_calls")):
            function_obj = _get_field(tool_call, "function")
            tool_name = _get_field(function_obj, "name") or _get_field(tool_call, "name")
            if tool_name:
                return str(tool_name)
        for message in _as_list(_get_field(response, "messages")):
            for content in _as_list(_get_field(message, "contents")):
                tool_name = _get_field(content, "name")
                if tool_name:
                    return str(tool_name)
        for choice in _as_list(_get_field(response, "choices")):
            for tool_call in _as_list(_get_field(_get_field(choice, "message"), "tool_calls")):
                function_obj = _get_field(tool_call, "function")
                tool_name = _get_field(function_obj, "name") or _get_field(tool_call, "name")
                if tool_name:
                    return str(tool_name)
    except Exception as exc:
        logger.debug(f"Error while extracting first tool name: {exc}")
    return None


def extract_assistant_message(arguments):
    try:
        messages = []
        status = get_status_code(arguments)
        if status not in ("success", "completed"):
            if arguments.get("exception") is not None:
                return get_exception_message(arguments)
            if hasattr(arguments.get("result"), "error"):
                return arguments["result"].error
            return None
        response = arguments.get("result")
        if response is None:
            return ""
        if hasattr(response, "tools") and isinstance(response.tools, list) and response.tools:
            if isinstance(response.tools[0], dict):
                tools = [
                    {
                        "tool_id": tool.get("id", ""),
                        "tool_name": tool.get("name", ""),
                        "tool_arguments": tool.get("arguments", ""),
                    }
                    for tool in response.tools
                ]
                messages.append({"tools": tools})
        if hasattr(response, "text") and response.text:
            messages.append({"assistant": response.text})
        if hasattr(response, "messages") and response.messages:
            for msg in _as_list(response.messages):
                if hasattr(msg, "contents") and msg.contents:
                    tools = []
                    text_parts = []
                    for content in msg.contents:
                        content_type = type(content).__name__
                        if content_type == "FunctionCallContent" or (
                            hasattr(content, "call_id") and hasattr(content, "name")
                        ):
                            tools.append(
                                {
                                    "tool_id": getattr(content, "call_id", ""),
                                    "tool_name": getattr(content, "name", ""),
                                    "tool_arguments": getattr(content, "arguments", ""),
                                }
                            )
                        elif hasattr(content, "text") and content.text:
                            text_parts.append(content.text)
                        elif hasattr(content, "value") and content.value:
                            text_parts.append(content.value)
                    if tools:
                        messages.append({"tools": tools})
                    if text_parts:
                        messages.append({"assistant": " ".join(text_parts)})
                elif hasattr(msg, "text") and msg.text:
                    messages.append({"assistant": msg.text})
                elif hasattr(msg, "content") and msg.content:
                    messages.append({"assistant": msg.content})
        if hasattr(response, "output") and isinstance(response.output, list) and response.output:
            output_tools = []
            for output_item in response.output:
                if getattr(output_item, "type", None) == "function_call":
                    output_tools.append(
                        {
                            "tool_id": getattr(output_item, "call_id", ""),
                            "tool_name": getattr(output_item, "name", ""),
                            "tool_arguments": getattr(output_item, "arguments", ""),
                        }
                    )
            if output_tools:
                messages.append({"tools": output_tools})
        if hasattr(response, "output_text") and response.output_text:
            role = response.role if hasattr(response, "role") else "assistant"
            messages.append({role: response.output_text})
        if hasattr(response, "choices") and response.choices:
            first_choice = response.choices[0]
            if hasattr(first_choice, "message"):
                role = getattr(first_choice.message, "role", "assistant")
                messages.append({role: first_choice.message.content})
        return get_json_dumps(messages[0]) if messages else ""
    except (IndexError, AttributeError) as exc:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(exc))
        return None


def agent_inference_type(arguments):
    try:
        response = arguments.get("result")
        if not _response_contains_tool_calls(response):
            return INFERENCE_TURN_END
        agent_prefix = get_value(AGENT_PREFIX_KEY)
        tool_name = _extract_first_tool_name(response) or ""
        if tool_name and agent_prefix and tool_name.startswith(agent_prefix):
            return INFERENCE_AGENT_DELEGATION
        return INFERENCE_TOOL_CALL
    except Exception as exc:
        logger.warning("Warning: Error occurred in agent_inference_type: %s", str(exc))
        return INFERENCE_TURN_END


def extract_finish_reason(arguments):
    try:
        if arguments.get("exception") is not None and hasattr(arguments["exception"], "code"):
            return arguments["exception"].code
        response = arguments.get("result")
        if not response:
            return None
        if _response_contains_tool_calls(response):
            return "tool_calls"
        direct_finish_reason = _get_field(response, "finish_reason")
        if direct_finish_reason:
            return direct_finish_reason.value if hasattr(direct_finish_reason, "value") else str(direct_finish_reason)
        choices = _as_list(_get_field(response, "choices"))
        if choices:
            choice_finish_reason = _get_field(choices[0], "finish_reason")
            if choice_finish_reason:
                return (
                    choice_finish_reason.value
                    if hasattr(choice_finish_reason, "value")
                    else str(choice_finish_reason)
                )
        if hasattr(response, "text") or hasattr(response, "messages") or hasattr(response, "output_text"):
            return "stop"
    except Exception as exc:
        logger.warning(f"Error extracting finish_reason: {exc}")
    return None

def extract_tool_name(arguments):
    return _extract_first_tool_name(arguments.get("result"))

def extract_tool_type(arguments):
    try:
        tool_name = extract_tool_name(arguments)
        if not tool_name:
            return None
        agent_prefix = get_value(AGENT_PREFIX_KEY)
        if agent_prefix and tool_name.startswith(agent_prefix):
            return "agent.microsoft"
        return "tool.microsoft"
    except Exception as exc:
        logger.warning(f"Error extracting tool type: {exc}")
    return None


def update_span_from_llm_response(response):
    meta_dict = {}
    try:
        if response is None:
            return meta_dict
        result = response.get("result") if isinstance(response, dict) else response
        if result is None:
            return meta_dict

        usage = _get_field(result, "usage")
        if isinstance(usage, dict):
            meta_dict["completion_tokens"] = usage.get("completion_tokens", 0)
            meta_dict["prompt_tokens"] = usage.get("prompt_tokens", 0)
            meta_dict["total_tokens"] = usage.get("total_tokens", 0)
            return meta_dict
        if usage is not None and hasattr(usage, "input_token_count"):
            meta_dict["completion_tokens"] = getattr(usage, "output_token_count", 0) or 0
            meta_dict["prompt_tokens"] = getattr(usage, "input_token_count", 0) or 0
            meta_dict["total_tokens"] = getattr(usage, "total_token_count", 0) or 0
            return meta_dict

        meta_dict.update({"completion_tokens": _get_field(_get_field(result, "usage_details"), "output_token_count", 0)})
        meta_dict.update({"prompt_tokens": _get_field(_get_field(result, "usage_details"), "input_token_count", 0)})
        meta_dict.update({"total_tokens": _get_field(_get_field(result, "usage_details"), "total_token_count", 0)})
        if meta_dict:
            return meta_dict
        if _response_contains_tool_calls(result):
            return {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
    except Exception as exc:
        logger.warning(f"Error updating span from LLM response: {exc}")
    return meta_dict