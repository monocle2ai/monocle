import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from monocle_apptrace.instrumentation.common.utils import (
    get_json_dumps,
    get_exception_message,
    get_status_code,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import map_azure_ai_inference_finish_reason_to_finish_type
from contextlib import suppress

logger = logging.getLogger(__name__)


def extract_messages(args_or_kwargs: Any) -> str:
    """Extract messages from azure-ai-inference request arguments."""
    try:
        messages = []
        if "instructions" in args_or_kwargs:
            messages.append({"instructions": args_or_kwargs.get("instructions", {})})
        if "input" in args_or_kwargs:
            messages.append({"input": args_or_kwargs.get("input", {})})
        if "messages" in args_or_kwargs and len(args_or_kwargs["messages"]) > 0:
            for msg in args_or_kwargs["messages"]:
                if msg.get("content") and msg.get("role"):
                    messages.append({msg["role"]: msg["content"]})

        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_inference_endpoint(instance: Any) -> str:
    """Extract inference endpoint from azure-ai-inference client instance."""
    try:
        return instance._config.endpoint
    except Exception as e:
        logger.warning(
            "Warning: Error occurred in extract_inference_endpoint: %s", str(e)
        )
        return ""


def extract_embeddings_input(args_or_kwargs: Any) -> str:
    """Extract input text from azure-ai-inference embeddings request."""
    try:
        # Handle both args and kwargs scenarios
        if isinstance(args_or_kwargs, dict):
            if "input" in args_or_kwargs:
                input_data = args_or_kwargs["input"]
            elif len(args_or_kwargs) > 0:
                first_arg = list(args_or_kwargs.values())[0]
                if isinstance(first_arg, dict) and "input" in first_arg:
                    input_data = first_arg["input"]
                else:
                    input_data = first_arg
            else:
                return ""
        elif hasattr(args_or_kwargs, "__iter__") and len(args_or_kwargs) > 0:
            first_arg = args_or_kwargs[0]
            if hasattr(first_arg, "get") and "input" in first_arg:
                input_data = first_arg["input"]
            else:
                input_data = first_arg
        else:
            return ""

        # Format input for display
        if isinstance(input_data, (list, tuple)):
            return " | ".join(str(item) for item in input_data)
        else:
            return str(input_data)
    except Exception as e:
        logger.warning(
            "Warning: Error occurred in extract_embeddings_input: %s", str(e)
        )
        return ""


def extract_assistant_message(arguments: Dict[str, Any]) -> str:
    """Extract assistant response from azure-ai-inference completion result."""
    try:
        # Check for exception first
        if arguments.get("exception") is not None:
            return get_exception_message(arguments)

        result = arguments.get("result")
        role = "assistant"
        messages = []
        if not result:
            return ""
        if hasattr(result, "output_text"):
            # If the result has output_text attribute
            role = getattr(result, "role", role)
            if "assistant" in role.lower():
                # If the role is assistant, we can assume it's a chat completion
                role = "assistant"
            messages.append({role: result.output_text})
        if (hasattr(result, "choices")
            and result.choices
            and result.choices[0].message
            and result.choices[0].message.content
        ):
            role = getattr(result.choices[0].message, "role", role)
            if "assistant" in role.lower():
                # If the role is assistant, we can assume it's a chat completion
                role = "assistant"
            # If the result is a chat completion with content
            messages.append({role: result.choices[0].message.content})
        return get_json_dumps(messages[0]) if messages else ""
    except Exception as e:
        logger.warning(
            "Warning: Error occurred in extract_assistant_message: %s", str(e)
        )
        return ""


def extract_embeddings_output(arguments: Dict[str, Any]) -> str:
    """Extract embeddings from azure-ai-inference embeddings result."""
    try:
        result = arguments.get("result")
        if not result:
            return ""

        if hasattr(result, "data") and result.data:
            # Format as summary of embeddings data
            embeddings_info = []
            for i, item in enumerate(result.data):
                if hasattr(item, "embedding") and hasattr(item, "index"):
                    embedding_length = len(item.embedding) if item.embedding else 0
                    embeddings_info.append(
                        f"index={item.index}, embedding=[{embedding_length} dimensions]"
                    )
            return " | ".join(embeddings_info)

        return str(result)
    except Exception as e:
        logger.warning(
            "Warning: Error occurred in extract_embeddings_output: %s", str(e)
        )
        return ""


def update_span_from_llm_response(result: Any, instance: Any = None) -> Dict[str, Any]:
    """Extract usage metadata from azure-ai-inference response."""
    try:
        attributes = {}

        # Handle streaming responses with accumulated usage data
        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            if hasattr(usage, "completion_tokens"):
                attributes["completion_tokens"] = usage.completion_tokens
            if hasattr(usage, "prompt_tokens"):
                attributes["prompt_tokens"] = usage.prompt_tokens
            if hasattr(usage, "total_tokens"):
                attributes["total_tokens"] = usage.total_tokens

        # Handle regular response usage
        elif hasattr(result, "usage"):
            usage = result.usage
            if hasattr(usage, "completion_tokens"):
                attributes["completion_tokens"] = usage.completion_tokens
            if hasattr(usage, "prompt_tokens"):
                attributes["prompt_tokens"] = usage.prompt_tokens
            if hasattr(usage, "total_tokens"):
                attributes["total_tokens"] = usage.total_tokens

        # Extract model information if available
        if hasattr(result, "model"):
            attributes["model"] = result.model

        return attributes
    except Exception as e:
        logger.warning(
            "Warning: Error occurred in update_span_from_llm_response: %s", str(e)
        )
        return {}


def get_model_name(arguments: Dict[str, Any]) -> str:
    """Extract model name from azure-ai-inference request arguments."""
    try:

        # Try to get from instance
        instance = arguments.get("instance")
        if arguments.get('kwargs') and arguments.get('kwargs').get('model'):
            return arguments['kwargs'].get('model')
        if instance and hasattr(instance, "_config") and hasattr(instance._config, "model"):
            return instance._config.endpoint.split("/")[-1]

        return ""
    except Exception as e:
        logger.warning("Warning: Error occurred in get_model_name: %s", str(e))
        return ""


def get_inference_type(arguments) -> str:
    instance = arguments.get("instance")
    if instance and hasattr(instance, "_config") and hasattr(instance._config, "endpoint"):
        endpoint = instance._config.endpoint
        try:
            parsed = urlparse(endpoint)
            hostname = parsed.hostname or endpoint
            hostname = hostname.lower()
        except Exception:
            hostname = str(endpoint).lower()
        if hostname.endswith("services.ai.azure.com"):
            return "azure_ai_inference"
        if hostname.endswith("openai.azure.com"):
            return "azure_openai"
    return "azure_ai_inference"


def get_provider_name(instance: Any) -> str:
    """Extract provider name from azure-ai-inference client instance."""
    try:
        # extract hostname from instance._config.endpoint
        # https://okahu-openai-dev.openai.azure.com/openai/deployments/kshitiz-gpt => okahu-openai-dev.openai.azure.com
        endpoint = instance._config.endpoint
        if endpoint:
            # Extract the hostname part
            provider_name = endpoint.split("/")[2] if "/" in endpoint else endpoint
            return provider_name
    except Exception as e:
        logger.warning("Warning: Error occurred in get_provider_name: %s", str(e))
        return "azure_ai_inference"


def extract_finish_reason(arguments: Dict[str, Any]) -> Optional[str]:
    """Extract finish_reason from Azure AI Inference response."""
    try:
        # Handle exception cases first
        if arguments.get("exception") is not None:
            ex = arguments["exception"]
            if hasattr(ex, "message") and isinstance(ex.message, str):
                message = ex.message
                if "content_filter" in message.lower():
                    return "content_filter"
            return "error"
        
        result = arguments.get("result")
        if result is None:
            return None
            
        # Check various possible locations for finish_reason in Azure AI Inference responses
        
        # Direct finish_reason attribute
        if hasattr(result, "finish_reason") and result.finish_reason:
            return result.finish_reason
            
        # Check for choices structure (OpenAI-compatible format)
        if hasattr(result, "choices") and result.choices:
            choice = result.choices[0]
            if hasattr(choice, "finish_reason") and choice.finish_reason:
                return choice.finish_reason
        
        # Check for additional metadata or response attributes
        if hasattr(result, "additional_kwargs") and result.additional_kwargs:
            kwargs = result.additional_kwargs
            if isinstance(kwargs, dict):
                for key in ["finish_reason", "stop_reason"]:
                    if key in kwargs:
                        return kwargs[key]
        
        # Check for response metadata
        if hasattr(result, "response_metadata") and result.response_metadata:
            metadata = result.response_metadata
            if isinstance(metadata, dict):
                for key in ["finish_reason", "stop_reason"]:
                    if key in metadata:
                        return metadata[key]
        
        # Check for streaming response with accumulated finish reason
        if hasattr(result, "type") and result.type == "stream":
            # For streaming responses, default to stop if completed successfully
            return "stop"
        
        # If no specific finish reason found, infer from status
        status_code = get_status_code(arguments)
        if status_code == 'success':
            return "stop"  # Default success finish reason
        elif status_code == 'error':
            return "error"
            
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None
    
    return None


def map_finish_reason_to_finish_type(finish_reason: Optional[str]) -> Optional[str]:
    """Map Azure AI Inference finish_reason to finish_type."""
    return map_azure_ai_inference_finish_reason_to_finish_type(finish_reason)

def _get_first_tool_call(response):
    """Helper function to extract the first tool call from various LangChain response formats"""
    with suppress(AttributeError, IndexError, TypeError):
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "tool_calls"):
                tool_calls = choice.message.tool_calls
                if tool_calls and len(tool_calls) > 0:
                    first_tool_call = tool_calls[0]
                    return first_tool_call

    return None

def extract_tool_name(arguments: Dict[str, Any]) -> Optional[str]:
    """Extract tool name from Azure AI Inference response when finish_type is tool_call."""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"])
        if not tool_call:
            return None

        for getter in [
            lambda tc: tc.function.name,  # dict with name key
            lambda tc: tc.name,
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s", str(e))
    
    return None

def extract_tool_type(arguments: Dict[str, Any]) -> Optional[str]:
    """Extract tool type from Azure AI Inference response when finish_type is tool_call."""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_name = extract_tool_name(arguments)
        if tool_name:
            return "tool.function"
            
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_type: %s", str(e))
    
    return None
