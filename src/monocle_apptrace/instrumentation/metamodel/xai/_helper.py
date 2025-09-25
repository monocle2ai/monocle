"""
This module provides utility functions for extracting system, user,
and assistant messages from xAI SDK inputs and outputs.
"""

import logging
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    try_option,
    get_exception_message,
    get_status_code,
)
from monocle_apptrace.instrumentation.common.span_handler import NonFrameworkSpanHandler
from monocle_apptrace.instrumentation.metamodel.finish_types import (
    map_openai_finish_reason_to_finish_type,
)
from monocle_apptrace.instrumentation.common.constants import INFERENCE_TURN_END

logger = logging.getLogger(__name__)

xai_roles_list = ['invalid', 'user', 'assistant', 'system', 'function', 'tool']

def extract_messages(kwargs):
    """Extract system and user messages from xAI chat.create() or chat.sample() or chat.append() calls"""
    try:
        messages = []
        
        # For chat.create() - extract from messages parameter
        if hasattr(kwargs['instance'], 'messages') and len(kwargs['instance'].messages) > 0:
            for msg in kwargs['instance'].messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # xAI message objects with role and content attributes
                    message_content = ""
                    for part in msg.content:
                        message_content += part.text
                    messages.append({xai_roles_list[msg.role]: message_content})
        
        # For chat.sample() - the chat object may contain message history
        if 'self' in kwargs:
            chat_instance = kwargs['self']
            if hasattr(chat_instance, 'messages') and chat_instance.messages:
                for msg in chat_instance.messages:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        messages.append({msg.role: msg.content})
        
        # Handle single message parameter
        if 'message' in kwargs:
            msg = kwargs['message']
            if isinstance(msg, str):
                messages.append({'user': msg})
            elif hasattr(msg, 'content'):
                role = getattr(msg, 'role', 'user')
                messages.append({role: msg.content})

        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(arguments):
    """Extract assistant response from xAI completion"""
    try:
        messages = []
        status = get_status_code(arguments)
        
        if status == 'success' or status == 'completed':
            response = arguments["result"]
            
            # Handle xAI chat response object
            if hasattr(response, "content"):
                messages.append({"assistant": response.content})
            
            # Handle streaming response
            if hasattr(response, "text"):
                messages.append({"assistant": response.text})
                
            # Handle response with role attribute
            if hasattr(response, "role") and hasattr(response, "content"):
                messages.append({response.role: response.content})
            
            # Handle chat instance after sample() call
            if hasattr(response, "messages") and response.messages:
                last_message = response.messages[-1]
                if hasattr(last_message, "role") and hasattr(last_message, "content"):
                    messages.append({last_message.role: last_message.content})
            
            return get_json_dumps(messages[0]) if messages else ""
        else:
            if arguments["exception"] is not None:
                return get_exception_message(arguments)
            elif hasattr(arguments["result"], "error"):
                return arguments["result"].error

    except (IndexError, AttributeError) as e:
        logger.warning(
            "Warning: Error occurred in extract_assistant_message: %s", str(e)
        )
        return None


def extract_provider_name(instance):
    """Extract provider name from xAI client"""
    try:
        # xAI SDK typically uses api.x.ai as endpoint
        if hasattr(instance, '_client') and hasattr(instance._client, 'base_url'):
            provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
            return provider_url.unwrap_or('x.ai')
        elif hasattr(instance, 'base_url'):
            provider_url: Option[str] = try_option(getattr, instance, 'base_url').map(str)
            return provider_url.unwrap_or('x.ai')
        return 'x.ai'
    except Exception:
        return 'x.ai'


def extract_inference_endpoint(instance):
    """Extract inference endpoint from xAI client"""
    try:
        if hasattr(instance, '_client') and hasattr(instance._client, 'api_host'):
            inference_endpoint: Option[str] = try_option(getattr, instance._client, 'api_host').map(str)
            return inference_endpoint.unwrap_or('https://api.x.ai/v1')
        elif hasattr(instance, 'base_url'):
            inference_endpoint: Option[str] = try_option(getattr, instance, 'base_url').map(str)
            return inference_endpoint.unwrap_or('https://api.x.ai/v1')
        return 'https://api.x.ai/v1'
    except Exception:
        return 'https://api.x.ai/v1'


def update_span_from_llm_response(response):
    """Update span metadata from xAI response"""
    meta_dict = {}
    try:
        # xAI responses may have usage information
        if response is not None and hasattr(response, "usage"):
            token_usage = response.usage
            if token_usage is not None:
                meta_dict.update({"completion_tokens": getattr(token_usage, "completion_tokens", None)})
                meta_dict.update({"prompt_tokens": getattr(token_usage, "prompt_tokens", None)})
                meta_dict.update({"total_tokens": getattr(token_usage, "total_tokens", None)})
        
        # Handle response metadata
        if hasattr(response, 'response_metadata'):
            response_metadata = response.response_metadata
            token_usage = response_metadata.get("token_usage")
            if token_usage is not None:
                meta_dict.update({"completion_tokens": getattr(token_usage, "completion_tokens", None)})
                meta_dict.update({"prompt_tokens": getattr(token_usage, "prompt_tokens", None)})
                meta_dict.update({"total_tokens": getattr(token_usage, "total_tokens", None)})
    except Exception as e:
        logger.warning("Warning: Error occurred in update_span_from_llm_response: %s", str(e))
    
    return meta_dict


def get_inference_type(instance):
    """Get inference type for xAI"""
    return 'xai'


class XAISpanHandler(NonFrameworkSpanHandler):
    """Span handler for xAI SDK instrumentation"""
    
    def skip_processor(self, to_wrap, wrapped, instance, span, args, kwargs) -> list[str]:
        return super().skip_processor(to_wrap, wrapped, instance, span, args, kwargs)

    def hydrate_events(self, to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=None, ex:Exception=None) -> bool:
        return super().hydrate_events(to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=parent_span, ex=ex)


def extract_finish_reason(arguments):
    """Extract finish_reason from xAI response"""
    try:
        if arguments["exception"] is not None:
            return "error"
            
        response = arguments["result"]
        
        # Handle xAI specific finish reasons
        if hasattr(response, "finish_reason") and response.finish_reason:
            return response.finish_reason
            
        # Default to stop for successful completion
        return "stop"
    except (IndexError, AttributeError) as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None


def map_finish_reason_to_finish_type(finish_reason):
    """Map xAI finish_reason to finish_type"""
    # Use OpenAI mapping as baseline since xAI follows similar patterns
    return map_openai_finish_reason_to_finish_type(finish_reason)


def agent_inference_type(arguments):
    """Extract agent inference type from xAI response"""
    # For now, xAI doesn't have built-in tool calling, so most are turn ends
    return INFERENCE_TURN_END