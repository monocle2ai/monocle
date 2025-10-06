"""
This module provides utility functions for extracting system, user,
and assistant messages from xAI SDK inputs and responses.
"""

import json
import logging
from monocle_apptrace.instrumentation.common.utils import (
    get_json_dumps,
    get_status_code,
)
from monocle_apptrace.instrumentation.common.span_handler import NonFrameworkSpanHandler

logger = logging.getLogger(__name__)

XAI_CHAT_ROLES = ['invalid', 'user', 'assistant','system', 'function', 'tool']

def extract_messages(kwargs):
    """Extract messages from xAI SDK request and return as JSON list"""
    try:
        messages = []
        
        # The xAI SDK stores messages in the chat instance's _proto.messages
        # We need to check the instance for the conversation history
        # For now, return an empty list since messages are in the chat instance
        # which would need to be passed differently
        
        return [get_json_dumps(msg) for msg in messages] if messages else []
        
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []

def extract_messages_from_instance(instance):
    """Extract messages from xAI SDK chat instance"""
    try:
        messages = []
        
        if hasattr(instance, '_proto') and hasattr(instance._proto, 'messages'):
            for msg in instance._proto.messages:
                # Extract role from protobuf enum
                if hasattr(msg, 'role'):
                    role = XAI_CHAT_ROLES[msg.role] if 0 <= msg.role < len(XAI_CHAT_ROLES) else 'unknown'
                else:
                    role = 'unknown'
                # Extract content from protobuf message
                content_parts = []
                for content_item in msg.content:
                    if hasattr(content_item, 'text'):
                        content_parts.append(content_item.text)
                
                content = ' '.join(content_parts) if content_parts else ''
                if content:
                    messages.append({role: content})
        
        return [get_json_dumps(msg) for msg in messages] if messages else []
        
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages_from_instance: %s", str(e))
        return []

def extract_assistant_message(arguments):
    """Extract assistant response from xAI SDK output"""
    try:
        messages = []
        status = get_status_code(arguments)
        
        if status == 'success' or status == 'completed':
            result = arguments.get("result")
            
            # xAI SDK Response object has .content property
            if hasattr(result, "content"):
                messages.append({"assistant": result.content})
            elif hasattr(result, "text"):
                messages.append({"assistant": result.text})
            elif isinstance(result, dict) and "content" in result:
                messages.append({"assistant": result["content"]})
            
            return get_json_dumps(messages[0]) if messages else ""
        else:
            # Handle error cases
            if arguments.get("exception") is not None:
                return get_exception_message(arguments)
                
        return ""
        
    except Exception as e:
        logger.warning(f"Error extracting assistant message: {e}")
        return ""

def extract_provider_name(instance):
    """Extract provider name from xAI SDK instance"""
    try:
        # xAI SDK uses gRPC clients, check for base channel or API host information
        if hasattr(instance, '_stub') and hasattr(instance._stub, '_channel'):
            # For local/development, return xai.local
            return "xai.api"
        return "xai.default"
    except Exception:
        return "xai.default"

def extract_inference_endpoint(instance):
    """Extract inference endpoint from xAI SDK instance"""
    try:
        # xAI SDK uses gRPC, so return the API endpoint
        return "https://api.x.ai"
    except Exception:
        return "https://api.x.ai"

def get_inference_type(instance):
    """Get inference type for xAI SDK"""
    return "xai"

class XaiSpanHandler(NonFrameworkSpanHandler):
    """Span handler for xAI SDK instrumentation"""
    
    def skip_processor(self, to_wrap, wrapped, instance, span, args, kwargs) -> list[str]:
        return super().skip_processor(to_wrap, wrapped, instance, span, args, kwargs)
    
    def hydrate_events(self, to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=None, ex: Exception=None) -> bool:
        return super().hydrate_events(to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=parent_span, ex=ex)
    
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

def extract_finish_reason(arguments):
    """Extract finish_reason from xAI SDK response"""
    result = arguments.get("result")
    if not result:
        return None
    
    try:
        # xAI SDK Response has .finish_reason property
        if hasattr(result, "finish_reason"):
            return result.finish_reason
            
    except Exception as e:
        logger.warning(f"Error extracting finish reason: {e}")
    
    return None

def map_finish_reason_to_finish_type(finish_reason):
    """Map xAI SDK finish_reason to standard finish_type"""
    if not finish_reason:
        return None
        
    # xAI SDK finish reason mapping
    xai_finish_mapping = {
        "FINISH_REASON_STOP": "stop",
        "FINISH_REASON_LENGTH": "length", 
        "FINISH_REASON_MAX_TOKENS": "length",
        "FINISH_REASON_TIMEOUT": "stop",
        "FINISH_REASON_ERROR": "error",
        "FINISH_REASON_TOOL_CALLS": "tool_calls"
    }
    
    return xai_finish_mapping.get(finish_reason, finish_reason)

def extract_usage(arguments):
    """Extract usage information from xAI SDK response"""
    result = arguments.get("result")
    if not result:
        return {}
    
    try:
        if hasattr(result, "usage"):
            usage = result.usage
            return {
                "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(usage, 'completion_tokens', 0), 
                "total_tokens": getattr(usage, 'total_tokens', 0)
            }
    except Exception as e:
        logger.warning(f"Error extracting usage: {e}")
    
    return {}

def agent_inference_type(arguments):
    """Extract agent inference type from xAI SDK response"""
    # TODO: Implement if xAI SDK supports agent/tool calling patterns
    # Check for tool calls in response
    result = arguments.get("result")
    if result and hasattr(result, "tool_calls") and result.tool_calls:
        return "INFERENCE_TOOL_CALL"
    return None

def get_exception_message(arguments):
    """Extract exception message from arguments"""
    exception = arguments.get("exception")
    if exception:
        return f"Error: {str(exception)}"
    return ""

# Helper validation for development
def validate_helper_functions():
    """Test helper functions independently during development"""
    
    print("🧪 Testing Helper Functions")
    
    # Test extract_messages
    test_kwargs = {
        "model": "grok-3-latest"
    }
    
    try:
        messages = extract_messages(test_kwargs)
        print(f"✅ extract_messages: {messages}")
        
        # Mock result for other tests
        class MockResponse:
            def __init__(self):
                self.content = "test response"
                self.finish_reason = "FINISH_REASON_STOP"
                self.usage = type('obj', (object,), {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15
                })()
        
        test_args = {
            "kwargs": test_kwargs,
            "result": MockResponse(),
            "instance": None
        }
        
        response = extract_assistant_message(test_args)
        print(f"✅ extract_assistant_message: {response}")
        
        provider = extract_provider_name(test_args.get("instance"))
        print(f"✅ extract_provider_name: {provider}")
        
        finish_reason = extract_finish_reason(test_args)
        print(f"✅ extract_finish_reason: {finish_reason}")
        
        usage = extract_usage(test_args)
        print(f"✅ extract_usage: {usage}")
        
    except Exception as e:
        print(f"❌ Helper function error: {e}")
        raise

# Create alias for backward compatibility
xai_handler = XaiSpanHandler()

if __name__ == "__main__":
    validate_helper_functions()
