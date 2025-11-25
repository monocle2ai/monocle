import logging

from monocle_apptrace.instrumentation.common.constants import TOOL_TYPE
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
    get_exception_message,
    get_status_code,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import map_haystack_finish_reason_to_finish_type
import json
from contextlib import suppress

logger = logging.getLogger(__name__)


def extract_messages(kwargs):
    try:
        messages = []
        system_message, user_message = None,None
        if isinstance(kwargs, dict):
            if 'system_prompt' in kwargs and kwargs['system_prompt']:
                system_message = kwargs['system_prompt']
            if 'prompt' in kwargs and kwargs['prompt']:
                user_message = extract_question_from_prompt(kwargs['prompt'])
            if 'messages' in kwargs and len(kwargs['messages'])>1:
                system_message = kwargs['messages'][0].text
                user_message = kwargs['messages'][1].text
            if system_message and user_message:
                messages.append({"system": system_message})
                messages.append({"user": user_message})

        return [str(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_question_from_prompt(content):
    try:
        question_prefix = "Question:"
        answer_prefix = "Answer:"

        question_start = content.find(question_prefix)
        if question_start == -1:
            return None  # Return None if "Question:" is not found

        question_start += len(question_prefix)
        answer_start = content.find(answer_prefix, question_start)
        if answer_start == -1:
            question = content[question_start:].strip()
        else:
            question = content[question_start:answer_start].strip()

        return question
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_question_from_prompt: %s", str(e))
        return ""

def extract_assistant_message(arguments):
    status = get_status_code(arguments)
    messages = []
    role = "assistant"
    if status == 'success':
        response = ""
        if "replies" in arguments['result']:
            reply = arguments['result']["replies"][0]
            if hasattr(reply, role) and hasattr(reply,role, "value") and isinstance(reply.role.value, str):
                role = reply.role.value or role
            if hasattr(reply, 'content'):
                response = reply.content
            elif hasattr(reply, 'text'):
                response = reply.text
            else:
                response = reply
        messages.append({role: response})
    else:
        if arguments["exception"] is not None:
            return get_exception_message(arguments)
        elif hasattr(arguments["result"], "error"):
            return arguments['result'].error
    return get_json_dumps(messages[0]) if messages else ""

def get_vectorstore_deployment(my_map):
    if isinstance(my_map, dict):
        if '_client_settings' in my_map:
            client = my_map['_client_settings'].__dict__
            host, port = get_keys_as_tuple(client, 'host', 'port')
            if host:
                return f"{host}:{port}" if port else host
        keys_to_check = ['client', '_client']
        host = get_host_from_map(my_map, keys_to_check)
        if host:
            return host
    else:
        if hasattr(my_map, 'client') and '_endpoint' in my_map.client.__dict__:
            return my_map.client.__dict__['_endpoint']
        host, port = get_keys_as_tuple(my_map.__dict__, 'host', 'port')
        if host:
            return f"{host}:{port}" if port else host
    return None


def get_host_from_map(my_map, keys_to_check):
    for key in keys_to_check:
        seed_connections = get_nested_value(my_map, [key, 'transport', 'seed_connections'])
        if seed_connections and 'host' in seed_connections[0].__dict__:
            return seed_connections[0].__dict__['host']
    return None


def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None

def extract_inference_endpoint(instance):
    if hasattr(instance, '_model_name') and isinstance(instance._model_name, str) and 'gemini' in instance._model_name.lower():
        inference_endpoint = try_option(lambda: f"https://generativelanguage.googleapis.com/v1beta/models/{instance._model_name}:generateContent")
    if hasattr(instance, 'client') and hasattr(instance.client, 'base_url'):
        inference_endpoint: Option[str] = try_option(getattr, instance.client, 'base_url').map(str)
    if inference_endpoint.is_none():
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(None)

def extract_embeding_model(instance):
    pipeline = try_option(getattr, instance, '__haystack_added_to_pipeline__')
    return pipeline.map(lambda p: try_option(getattr, p, 'get_component').map(
        lambda g: try_option(getattr, g('text_embedder'), 'model').unwrap_or(None)).unwrap_or(None)).unwrap_or(None)

def update_span_from_llm_response(response, instance):
    meta_dict = {}
    token_usage = None
    if response is not None and isinstance(response, dict):
        if "meta" in response:
            token_usage = response["meta"][0]["usage"]
        elif "replies" in response:  # and "meta" in response["replies"][0]:
            token_usage = response["replies"][0].meta["usage"]
        if token_usage is not None:
            temperature = instance.__dict__.get("temperature", None)
            meta_dict.update({"temperature": temperature})
            meta_dict.update(
                {"completion_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens")})
            meta_dict.update({"prompt_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens")})
            meta_dict.update({"total_tokens": token_usage.get("total_tokens") or token_usage.get("completion_tokens")+token_usage.get("prompt_tokens")})
    return meta_dict


def update_output_span_events(results):
    output_arg_text = " ".join([doc.content for doc in results['documents']])
    if len(output_arg_text) > 100:
        output_arg_text = output_arg_text[:100] + "..."
    return output_arg_text

def extract_finish_reason(arguments):
    """Extract finish_reason from Haystack response."""
    try:
        # Handle exception cases first
        if arguments.get("exception") is not None:
            return "error"

        response = arguments.get("result")
        if response is None:
            return None

        # Direct finish_reason attribute
        if hasattr(response, "finish_reason") and response.finish_reason:
            return response.finish_reason

        if isinstance(response,dict) and 'meta' in response and response['meta'] and len(response['meta']) > 0:
            metadata = response['meta'][0]
            if isinstance(metadata, dict):
                # Check for finish_reason in metadata
                if "finish_reason" in metadata:
                    return metadata["finish_reason"]

        if isinstance(response,dict) and 'replies' in response and response['replies'] and len(response['replies']) > 0:
            metadata = response['replies'][0]
            if hasattr(metadata,'meta') and metadata.meta:
                if "finish_reason" in metadata.meta:
                    return metadata.meta["finish_reason"]

        # Check if response has generation_info
        if hasattr(response, "generation_info") and response.generation_info:
            finish_reason = response.generation_info.get("finish_reason")
            if finish_reason:
                return finish_reason

        # Check if response has llm_output (batch responses)
        if hasattr(response, "llm_output") and response.llm_output:
            finish_reason = response.llm_output.get("finish_reason")
            if finish_reason:
                return finish_reason

        # For AIMessage responses, check additional_kwargs
        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            finish_reason = response.additional_kwargs.get("finish_reason")
            if finish_reason:
                return finish_reason

        # For generation responses with choices (similar to OpenAI structure)
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "finish_reason"):
                return choice.finish_reason

        # Fallback: if no finish_reason found, default to "stop" (success)
        return "stop"
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None

def map_finish_reason_to_finish_type(finish_reason):
    """Map Haystack finish_reason to finish_type using Haystack mapping."""
    return map_haystack_finish_reason_to_finish_type(finish_reason)

def _get_first_tool_call(response, instance):
    """Helper function to extract the first tool call from various LangChain response formats"""

    with suppress(AttributeError, IndexError, TypeError):
        if isinstance(response, dict) and 'replies' in response and response['replies']:
            for reply in response['replies']:
                if hasattr(reply, 'content') and reply.content:
                    content_data = json.loads(reply.content)
                    if isinstance(content_data, dict) and 'type' in content_data:
                        if content_data.get('type') == 'tool_use':
                            return content_data

        if hasattr(instance, 'generation_kwargs') and instance.generation_kwargs:
            tools = instance.generation_kwargs.get('tools', [])
            if tools and len(tools) > 0:
                return tools[0]

    return None

def extract_tool_name(arguments):
    """Extract tool name from Haystack response when finish_type is tool_call"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"],arguments["instance"])
        if not tool_call:
            return None

        for getter in [
            lambda tc: tc["name"],
            lambda tc: tc["function"]["name"]
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s", str(e))
    
    return None

def extract_tool_type(arguments):
    """Extract tool type from Haystack response when finish_type is tool_call"""
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
