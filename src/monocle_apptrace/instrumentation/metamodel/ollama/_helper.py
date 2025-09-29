"""
This module provides utility functions for extracting system, user,
and assistant messages from Ollama inputs and responses.
"""

import json
import logging
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    try_option,
    get_exception_message,
    get_parent_span,
    get_status_code,
)
from monocle_apptrace.instrumentation.common.span_handler import NonFrameworkSpanHandler, WORKFLOW_TYPE_MAP
from monocle_apptrace.instrumentation.metamodel.finish_types import (
    map_openai_finish_reason_to_finish_type,
    OPENAI_FINISH_REASON_MAPPING
)
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, CHILD_ERROR_CODE, INFERENCE_AGENT_DELEGATION, INFERENCE_TURN_END, INFERENCE_TOOL_CALL

logger = logging.getLogger(__name__)

def extract_messages(kwargs):
    """Extract messages from Ollama chat request and return as JSON list like OpenAI"""
    try:
        messages = []
        
        # Handle 'prompt' for generate method
        if 'prompt' in kwargs:
            messages.append({'user': kwargs['prompt']})
        
        # Handle 'messages' for chat method
        if 'messages' in kwargs and len(kwargs['messages']) > 0:
            for msg in kwargs['messages']:
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role and content:
                        messages.append({role: content})
                else:
                    # Handle Message objects
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        messages.append({msg.role: msg.content})
        
        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []

def extract_assistant_message(arguments):
    """Extract assistant message from Ollama response (OpenAI-style approach)"""
    try:
        from monocle_apptrace.instrumentation.common.utils import get_json_dumps, get_status_code
        
        messages = []
        status = get_status_code(arguments)
        if status == 'success' or status == 'completed':
            result = arguments.get("result")
            
            # Handle streaming result (span result object with output_text)
            if hasattr(result, "output_text") and result.output_text:
                role = getattr(result, "role", "assistant")
                messages.append({role: result.output_text})
                return get_json_dumps(messages[0])
            
            # Handle regular ChatResponse
            elif hasattr(result, "message") and hasattr(result.message, "content"):
                role = getattr(result.message, "role", "assistant")
                messages.append({role: result.message.content})
                return get_json_dumps(messages[0])
                
            # Handle GenerateResponse  
            elif hasattr(result, "response"):
                messages.append({"assistant": result.response})
                return get_json_dumps(messages[0])
                
            # Handle dict response
            elif isinstance(result, dict):
                if "message" in result and "content" in result["message"]:
                    role = result["message"].get("role", "assistant")
                    messages.append({role: result["message"]["content"]})
                    return get_json_dumps(messages[0])
                elif "response" in result:
                    messages.append({"assistant": result["response"]})
                    return get_json_dumps(messages[0])
        else:
            if arguments.get("exception") is not None:
                from monocle_apptrace.instrumentation.common.utils import get_exception_message
                return get_exception_message(arguments)
            elif hasattr(arguments.get("result"), "error"):
                return arguments["result"].error
                
        return get_json_dumps(messages[0]) if messages else ""
        
    except Exception as e:
        logger.warning(f"Error extracting assistant message: {e}")
        return ""

def extract_provider_name(instance):
    """Extract provider name from Ollama instance"""
    try:
        if hasattr(instance, "_client") and hasattr(instance._client, "_base_url"):
            base_url = str(instance._client._base_url)
            if "localhost" in base_url or "127.0.0.1" in base_url:
                return "ollama.local"
            return base_url
    except Exception:
        pass
    return "ollama.local"

def extract_inference_endpoint(instance):
    """Extract inference endpoint from Ollama instance"""
    try:
        if hasattr(instance, "_client") and hasattr(instance._client, "_base_url"):
            return str(instance._client._base_url)
    except Exception:
        pass
    return "http://localhost:11434/"

def update_input_span_events(kwargs):
    """Update input span events with Ollama-specific formatting"""
    # This function is deprecated - use extract_messages directly
    return extract_messages(kwargs)

def update_output_span_events(results):
    """Update output span events with Ollama response"""
    try:
        if hasattr(results, "message") and hasattr(results.message, "content"):
            return results.message.content
        elif hasattr(results, "response"):
            return results.response
        elif isinstance(results, dict):
            if "message" in results and "content" in results["message"]:
                return results["message"]["content"]
            elif "response" in results:
                return results["response"]
    except Exception as e:
        logger.warning(f"Error updating output span events: {e}")
    
    return str(results) if results else ""

def update_span_from_llm_response(response):
    """Update span with metadata from Ollama response"""
    metadata = {}
    
    try:
        # Extract usage information if available
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                metadata["input_tokens"] = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                metadata["output_tokens"] = usage.completion_tokens
            if hasattr(usage, "total_tokens"):
                metadata["total_tokens"] = usage.total_tokens
                
        # Extract model information
        if hasattr(response, "model"):
            metadata["model"] = response.model
            
    except Exception as e:
        logger.warning(f"Error extracting metadata from Ollama response: {e}")
    
    return metadata

def extract_vector_input(vector_input: dict):
    """Extract vector input for embeddings"""
    if isinstance(vector_input, dict):
        return vector_input.get("input", vector_input.get("prompt", ""))
    return str(vector_input)

def extract_vector_output(vector_output):
    """Extract vector output from embeddings response"""
    try:
        if hasattr(vector_output, "embedding"):
            return vector_output.embedding
        elif hasattr(vector_output, "embeddings") and vector_output.embeddings:
            return vector_output.embeddings[0] if vector_output.embeddings else []
        elif isinstance(vector_output, dict):
            if "embedding" in vector_output:
                return vector_output["embedding"]
            elif "embeddings" in vector_output and vector_output["embeddings"]:
                return vector_output["embeddings"][0]
    except Exception as e:
        logger.warning(f"Error extracting vector output: {e}")
    return []

def get_inference_type(instance):
    """Get inference type for Ollama"""
    return "ollama"

class OllamaSpanHandler(NonFrameworkSpanHandler):
    """Span handler for Ollama instrumentation"""
    
    def skip_processor(self, to_wrap, wrapped, instance, span, args, kwargs) -> list[str]:
        return super().skip_processor(to_wrap, wrapped, instance, span, args, kwargs)
    
    def hydrate_events(self, to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=None, ex: Exception=None) -> bool:
        return super().hydrate_events(to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=parent_span, ex=ex)
    
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

def extract_finish_reason(arguments):
    """Extract finish_reason from Ollama response"""
    result = arguments.get("result")
    if not result:
        return None
    
    try:
        # Check for done_reason in response
        if hasattr(result, "done_reason"):
            return result.done_reason
        elif isinstance(result, dict) and "done_reason" in result:
            return result["done_reason"]
            
        # Check message for finish reason
        if hasattr(result, "message"):
            message = result.message
            if hasattr(message, "finish_reason"):
                return message.finish_reason
        elif isinstance(result, dict) and "message" in result:
            message = result["message"]
            if isinstance(message, dict) and "finish_reason" in message:
                return message["finish_reason"]
                
    except Exception as e:
        logger.warning(f"Error extracting finish reason: {e}")
    
    return None

def map_finish_reason_to_finish_type(finish_reason):
    """Map Ollama finish_reason to finish_type"""
    if not finish_reason:
        return None
        
    # Ollama uses different finish reasons than OpenAI
    ollama_finish_mapping = {
        "stop": "stop",
        "length": "length", 
        "model_length": "length",
        "abort": "stop",
        "cancelled": "stop"
    }
    
    return ollama_finish_mapping.get(finish_reason, finish_reason)

def agent_inference_type(arguments):
    """Extract agent inference type from Ollama response"""
    # Ollama doesn't have built-in agent patterns like OpenAI
    return None
