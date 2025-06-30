import logging
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
    get_exception_message,
    get_exception_status_code
)
import json
from typing import Any, Dict

logger = logging.getLogger(__name__)

def capture_input(arguments):
    """
    Captures the input from Teams AI state.
    Args:
        arguments (dict): Arguments containing state and context information
    Returns:
        str: The input message or error message
    """
    try:
        # Get the memory object from kwargs
        kwargs = arguments.get("kwargs", {})
        messages = []

        # If memory exists, try to get the input from temp
        if "memory" in kwargs:
            memory = kwargs["memory"]
            # Check if it's a TurnState object
            if hasattr(memory, "get"):
                # Use proper TurnState.get() method
                temp = memory.get("temp")
                if temp and hasattr(temp, "get"):
                    input_value = temp.get("input")
                    if input_value:
                        messages.append({'user': str(input_value)})
        system_prompt = ""
        try:
            system_prompt = kwargs.get("template").prompt.sections[0].sections[0].template
            messages.append({'system': system_prompt})
        except Exception as e:
            print(f"Debug - Error accessing system prompt: {str(e)}")
            
        # Try alternative path through context if memory path fails
        context = kwargs.get("context")
        if hasattr(context, "activity") and hasattr(context.activity, "text"):
            messages.append({'user': str(context.activity.text)})

        return [str(message) for message in messages]
    except Exception as e:
        print(f"Debug - Arguments structure: {str(arguments)}")
        print(f"Debug - kwargs: {str(kwargs)}")
        if "memory" in kwargs:
            print(f"Debug - memory type: {type(kwargs['memory'])}")
        return f"Error capturing input: {str(e)}"

def capture_prompt_info(arguments):
    """Captures prompt information from ActionPlanner state"""
    try:
        kwargs = arguments.get("kwargs", {})
        prompt = kwargs.get("prompt")

        if isinstance(prompt, str):
            return prompt
        elif hasattr(prompt, "name"):
            return prompt.name

        return "No prompt information found"
    except Exception as e:
        return f"Error capturing prompt: {str(e)}"

def capture_prompt_template_info(arguments):
    """Captures prompt information from ActionPlanner state"""
    try:
        kwargs = arguments.get("kwargs", {})
        prompt = kwargs.get("prompt")

        if hasattr(prompt,"prompt") and prompt.prompt is not None:
            if "_text" in prompt.prompt.__dict__:
                prompt_template = prompt.prompt.__dict__.get("_text", None)
                return prompt_template
            elif "_sections" in prompt.prompt.__dict__ and prompt.prompt._sections is not None:
                sections = prompt.prompt._sections[0].__dict__.get("_sections", None)
                if sections is not None and "_template" in sections[0].__dict__:
                    return sections[0].__dict__.get("_template", None)


        return "No prompt information found"
    except Exception as e:
        return f"Error capturing prompt: {str(e)}"

def status_check(arguments):
    if hasattr(arguments["result"], "error") and arguments["result"].error is not None:
        error_msg:str = arguments["result"].error
        error_code:str = arguments["result"].status if hasattr(arguments["result"], "status") else "unknown"
        raise MonocleSpanException(f"Error: {error_code} - {error_msg}")

def get_prompt_template(arguments):
    pass
    return {
        "prompt_template_name": capture_prompt_info(arguments),
        "prompt_template": capture_prompt_template_info(arguments),
        "prompt_template_description": get_nested_value(arguments.get("kwargs", {}),
                                                        ["prompt", "config", "description"]),
        "prompt_template_type": get_nested_value(arguments.get("kwargs", {}), ["prompt", "config", "type"])
    }

def get_status_code(arguments):
    if arguments["exception"] is not None:
        return get_exception_status_code(arguments)
    elif hasattr(arguments["result"], "status"):
        return arguments["result"].status
    else:
        return 'success'

def get_status(arguments):
    if arguments["exception"] is not None:
        return 'error'
    elif get_status_code(arguments) == 'success':
        return 'success'
    else:
        return 'error'
    
def get_response(arguments) -> str:
    status = get_status_code(arguments)
    response:str = ""
    if status == 'success':
        if hasattr(arguments["result"], "message"):
            response = arguments["result"].message.content 
        else:
            response = str(arguments["result"])
    else:
        if arguments["exception"] is not None:
            response = get_exception_message(arguments)
        elif hasattr(arguments["result"], "error"):
            response = arguments["result"].error
    return response

def check_status(arguments):
    status = get_status_code(arguments)
    if status != 'success':
        raise MonocleSpanException(f"{status}")   

def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))

def extract_search_endpoint(instance):
    if hasattr(instance, '_endpoint') and instance._endpoint is not None:
        return instance._endpoint
    else:
        return None

def extract_index_name(instance):
    if hasattr(instance, '_index_name') and instance._index_name is not None:
        return instance._index_name
    else:
        return None

def capture_vector_queries(kwargs):
    try:
        result = {}
        if 'vector_queries' in kwargs and kwargs['vector_queries'] is not None:
            vector_queries = kwargs['vector_queries'][0].__dict__
            for key, value in vector_queries.items():
                if key != 'vector':
                    result[key] = "null" if value is None else value
            return str(result)
    except Exception as e:
        print(f"Debug - Error capturing vector queries: {str(e)}")

def search_input(arguments):
    parameters = {
        "search_text": get_nested_value(arguments.get("kwargs", {}), ["search_text"]),
        "select": get_nested_value(arguments.get("kwargs", {}), ["select"]),
        "vector_queries": capture_vector_queries(arguments["kwargs"])
    }
    return _json(parameters)

def search_output(arguments):
    try:
        if hasattr(arguments["result"], "_args") and len(arguments["result"]._args) > 1:
            if hasattr(arguments["result"]._args[1], "request") and arguments["result"]._args[1].request is not None:
                request = arguments["result"]._args[1].request
                summary = {
                    "count": "null" if request.include_total_result_count is None else request.include_total_result_count,
                    "coverage": "null" if request.minimum_coverage is None else request.minimum_coverage,
                    "facets": "null" if request.facets is None else request.facets,
                }
                return _json(summary)
    except Exception as e:
        print(f"Debug - Error capturing facets: {str(e)}")
    return None

def capture_metadata(arguments) -> str:
    inst = arguments.get("instance", None)
    meta = {
        "endpoint": getattr(inst, "_endpoint", "unknown"),
        "index": getattr(inst, "_index_name", "unknown"),
        "latency_ms": arguments.get("latency_ms", "null"),
    }
    return _json(meta)


def _json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return str(value)

def search_post_input(kwargs):
    try:
        if 'search_request' in kwargs and kwargs['search_request'] is not None:
            req = kwargs['search_request']
            if hasattr(req, "search_text"):
                return getattr(req, "search_text", "")
    except Exception as e:
        print(f"Debug - Error capturing search input: {str(e)}")

def search_post_output(result):
    try:
        if not result or not getattr(result, "results", None):
            return "[]"
        filtered = []
        for item in result.results:
            doc = item.additional_properties or {}
            filtered.append(
                {
                    "docTitle": doc.get("docTitle"),
                    "description": doc.get("description"),
                    "@search.score": item.score,
                    "@search.reranker_score": item.reranker_score,
                }
            )
        return _json(filtered)
    except Exception as e:
        print(f"Debug - Error capturing search output: {str(e)}")
        return "[]"

def search_post_capture_meta(arguments):
    if 'search_request' in arguments['kwargs']:
        req = arguments['kwargs']['search_request']
    else:
        return "{}"
    wanted = [
        "select", "include_total_result_count", "facets", "filter",
        "highlight_fields", "highlight_post_tag", "highlight_pre_tag",
        "minimum_coverage", "order_by", "query_type", "scoring_parameters",
        "scoring_profile", "semantic_query",
    ]
    meta = {k: getattr(req, k, None) for k in wanted}
    meta["latency_ms"] = arguments.get("latency_ms")
    return _json(meta)