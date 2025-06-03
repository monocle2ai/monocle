import logging
from typing import Any, Dict, Optional
from monocle_apptrace.instrumentation.common.utils import (
    resolve_from_alias,
    get_exception_message,
)

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

        return [str(message) for message in messages]
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
        if not result:
            return ""
        if hasattr(result, "output_text"):
            # If the result has output_text attribute
            return result.output_text
        if (
            result.choices
            and result.choices[0].message
            and result.choices[0].message.content
        ):
            # If the result is a chat completion with content
            return result.choices[0].message.content

        return str(result)
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
    if arguments.get("instance") and arguments["instance"]._config.endpoint:
        
        endpoint = arguments["instance"]._config.endpoint
        if "services.ai.azure.com" in endpoint:
            return "azure_ai_inference"
        if "openai.azure.com" in endpoint:
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
