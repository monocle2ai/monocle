import logging
from contextlib import suppress
from monocle_apptrace.instrumentation.common.constants import TOOL_TYPE
from monocle_apptrace.instrumentation.common.utils import (
    get_exception_message,
    get_json_dumps,
    get_status_code,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import (
    map_gemini_finish_reason_to_finish_type,
    GEMINI_FINISH_REASON_MAPPING
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
            messages.append({'user': contents})

        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []

def extract_assistant_message(arguments):
    try:
        status = get_status_code(arguments)
        messages = []
        role = "model"
        if hasattr(arguments['result'], "candidates") and len(arguments['result'].candidates) > 0 and hasattr(arguments['result'].candidates[0], "content") and hasattr(arguments['result'].candidates[0].content, "role"):
                role = arguments["result"].candidates[0].content.role
        if status == 'success':
            if arguments["result"].parts[0].function_call is not None:
                role = "ai"
                messages.append({role: f'"model": {arguments["result"].parts[0].function_call.name}, "args": {arguments["result"].parts[0].function_call.args}'})
            elif hasattr(arguments['result'], "text") and len(arguments['result'].text):
                messages.append({role: arguments['result'].text})
        else:
            if arguments["exception"] is not None:
                return get_exception_message(arguments)
            elif hasattr(arguments["result"], "error"):
                return arguments["result"].error
        return get_json_dumps(messages[0]) if messages else ""
    except (IndexError, AttributeError) as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return None

def update_input_span_events(kwargs):
    if 'contents' in kwargs and isinstance(kwargs['contents'], list) and len(kwargs['contents']) > 0:
        query = kwargs['contents'][0]
        return query

def update_output_span_events(results):
    if hasattr(results,'embeddings') and isinstance(results.embeddings, list) and len(results.embeddings) > 0:
        embeddings = results.embeddings[0]
        if hasattr(embeddings, 'values') and isinstance(embeddings.values, list) and len(embeddings.values) > 100:
            output = str(results.embeddings[0].values[:100]) + "..."
            return output

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

def extract_finish_reason(arguments):
    """Extract finish_reason from Gemini response"""
    try:
        if "exception" in arguments and arguments["exception"] is not None:
            return None
            
        response = arguments["result"]

        with suppress(IndexError, AttributeError):
            if response.parts is not None and response.parts[0].function_call is not None:
                return "FUNCTION_CALL"
        
        # Handle Gemini response structure
        if (response is not None and 
            hasattr(response, "candidates") and 
            len(response.candidates) > 0 and 
            hasattr(response.candidates[0], "finish_reason")):
            return response.candidates[0].finish_reason
            
    except (IndexError, AttributeError) as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None
    return None

def map_finish_reason_to_finish_type(finish_reason):
    """Map Gemini finish_reason to finish_type based on the possible errors mapping"""
    return map_gemini_finish_reason_to_finish_type(finish_reason)

def _get_first_tool_call(response):

    with suppress(AttributeError, IndexError, TypeError):
        if hasattr(response,"parts") and len(response.parts)>0:
            for part in response.parts:
                if hasattr(part,"function_call") and part.function_call is not None:
                    return part.function_call

    return None

def extract_tool_name(arguments):
    """Extract tool name from Gemini response when function calls are present"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"])
        if not tool_call:
            return None

        for getter in [
            lambda tc: tc.name,
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s", str(e))

    return None

def extract_tool_type(arguments):
    """Extract tool type from Gemini response when function calls are present"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_name = extract_tool_name(arguments)
        if tool_name:
            return TOOL_TYPE

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_type: %s", str(e))

    return None
