import logging

from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
    get_exception_message,
    get_status_code,
)
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
    response: str = ""
    if status == 'success':
        if "replies" in arguments['result']:
            reply = arguments['result']["replies"][0]
            if hasattr(reply, 'content'):
                response = reply.content
            elif hasattr(reply, 'text'):
                response = reply.text
            else:
                response = reply
    else:
        if arguments["exception"] is not None:
            response = get_exception_message(arguments)
        elif hasattr(response, "error"):
            response = arguments['result'].error

    return response


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
