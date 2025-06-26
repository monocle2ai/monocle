import logging
from monocle_apptrace.instrumentation.common.utils import (
    get_exception_message,
    get_status_code,
)

logger = logging.getLogger(__name__)

def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None

def extract_messages(kwargs):
    """Extract system and user messages"""
    try:
        messages = []
        config = kwargs.get('config')
        if config and hasattr(config, 'system_instruction'):
            system_instructions = getattr(config, 'system_instruction', None)
            if system_instructions:
                messages.append({'system': system_instructions})

        contents = kwargs.get('contents')
        if isinstance(contents, list):
            for content in contents:
                if hasattr(content, 'parts') and getattr(content, 'parts'):
                    part = content.parts[0]
                    if hasattr(part, 'text'):
                        messages.append({getattr(content, 'role', 'user'): part.text})
        elif isinstance(contents, str):
            messages.append({'input': contents})

        return [str(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []

def extract_assistant_message(arguments):
    try:
        status = get_status_code(arguments)
        response: str = ""
        if status == 'success':
            if hasattr(arguments['result'], "text") and len(arguments['result'].text):
                response = arguments['result'].text
        else:
            if arguments["exception"] is not None:
                response = get_exception_message(arguments)
            elif hasattr(arguments["result"], "error"):
                response = arguments["result"].error
        return response
    except (IndexError, AttributeError) as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return None

def extract_inference_endpoint(instance):
    try:
        if hasattr(instance,'_api_client') and hasattr(instance._api_client, '_http_options'):
            if hasattr(instance._api_client._http_options,'base_url'):
                return instance._api_client._http_options.base_url
    except Exception as e:
        logger.warning("Warning: Error occurred in inference endpoint: %s", str(e))
        return []

def update_span_from_llm_response(response, instance):
    meta_dict = {}
    if response is not None and hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        token_usage = response.usage_metadata
        if token_usage is not None:
            meta_dict.update({"completion_tokens": token_usage.candidates_token_count})
            meta_dict.update({"prompt_tokens": token_usage.prompt_token_count })
            meta_dict.update({"total_tokens": token_usage.total_token_count})
    return meta_dict
