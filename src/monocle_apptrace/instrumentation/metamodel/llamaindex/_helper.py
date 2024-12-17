"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
from urllib.parse import urlparse
from opentelemetry.sdk.trace import Span
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
)
from monocle_apptrace.instrumentation.common.base_task_processor import TaskProcessor
from importlib.metadata import version
from monocle_apptrace.instrumentation.common.constants import (
    DATA_INPUT_KEY,
    DATA_OUTPUT_KEY,
    PROMPT_INPUT_KEY,
    PROMPT_OUTPUT_KEY,
    QUERY,
    RESPONSE,
    SESSION_PROPERTIES_KEY,
    INFRA_SERVICE_KEY,
    META_DATA
)

logger = logging.getLogger(__name__)


def extract_messages(args):
    """Extract system and user messages"""
    try:
        messages = []
        if isinstance(args[0], list):
            for msg in args[0]:
                if hasattr(msg, 'content') and hasattr(msg, 'role'):
                    role = getattr(msg.role, 'value', msg.role)
                    if role == "system":
                        messages.append({role: msg.content})
                    elif role in ["user", "human"]:
                        user_message = extract_query_from_content(msg.content)
                        messages.append({role: user_message})
        return [str(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(response):
    try:
        if isinstance(response, str):
            return [response]
        if hasattr(response, "content"):
            return [response.content]
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return [response.message.content]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return []


def extract_query_from_content(content):
    try:
        query_prefix = "Query:"
        answer_prefix = "Answer:"
        query_start = content.find(query_prefix)
        if query_start == -1:
            return None

        query_start += len(query_prefix)
        answer_start = content.find(answer_prefix, query_start)
        if answer_start == -1:
            query = content[query_start:].strip()
        else:
            query = content[query_start:answer_start].strip()
        return query
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_query_from_content: %s", str(e))
        return ""


def extract_provider_name(instance):
    provider_url = try_option(getattr, instance, 'api_base').and_then(lambda url: urlparse(url).hostname)
    return provider_url


def extract_inference_endpoint(instance):
    inference_endpoint = try_option(getattr, instance._client.sdk_configuration, 'server_url').map(str)
    return inference_endpoint.unwrap_or(extract_provider_name(instance))


def extract_vectorstore_deployment(my_map):
    if isinstance(my_map, dict):
        if '_client_settings' in my_map:
            client = my_map['_client_settings'].__dict__
            host, port = get_keys_as_tuple(client, 'host', 'port')
            if host:
                return f"{host}:{port}" if port else host
        keys_to_check = ['client', '_client']
        host = __get_host_from_map(my_map, keys_to_check)
        if host:
            return host
    else:
        if hasattr(my_map, 'client') and '_endpoint' in my_map.client.__dict__:
            return my_map.client.__dict__['_endpoint']
        host, port = get_keys_as_tuple(my_map.__dict__, 'host', 'port')
        if host:
            return f"{host}:{port}" if port else host
    return None


def __get_host_from_map(my_map, keys_to_check):
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


def is_root_span(curr_span: Span) -> bool:
    return curr_span.parent is None


def update_spans_with_data_input(to_wrap, wrapped_args, span: Span):
    package_name: str = to_wrap.get('package')
    if "retriever" in package_name and len(wrapped_args) > 0:
        input_arg_text = wrapped_args[0].query_str
        if input_arg_text:
            span.add_event(DATA_INPUT_KEY, {QUERY: input_arg_text})
    if is_root_span(span):
        input_arg_text = wrapped_args[0]
        if isinstance(input_arg_text, dict):
            span.add_event(PROMPT_INPUT_KEY, input_arg_text)
        else:
            span.add_event(PROMPT_INPUT_KEY, {QUERY: input_arg_text})


def update_spans_with_data_output(to_wrap, return_value, span: Span):
    package_name: str = to_wrap.get('package')
    if "retriever" in package_name:
        output_arg_text = return_value[0].text
        if len(output_arg_text) > 100:
            output_arg_text = output_arg_text[:100] + "..."
        span.add_event(DATA_OUTPUT_KEY, {RESPONSE: output_arg_text})
    if is_root_span(span):
        if isinstance(return_value, str):
            span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: return_value})
        elif isinstance(return_value, dict):
            span.add_event(PROMPT_OUTPUT_KEY, return_value)
        else:
            span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: return_value.response})


def update_span_from_llm_response(response, span: Span, instance):
    if response is not None and hasattr(response, "raw"):
        try:
            meta_dict = {}
            if response.raw is not None:
                token_usage = response.raw.get("usage") if isinstance(response.raw, dict) else getattr(response.raw,
                                                                                                       "usage", None)
                if token_usage is not None:
                    temperature = instance.__dict__.get("temperature", None)
                    meta_dict.update({"temperature": temperature})
                    if getattr(token_usage, "completion_tokens", None):
                        meta_dict.update({"completion_tokens": getattr(token_usage, "completion_tokens")})
                    if getattr(token_usage, "prompt_tokens", None):
                        meta_dict.update({"prompt_tokens": getattr(token_usage, "prompt_tokens")})
                    if getattr(token_usage, "total_tokens", None):
                        meta_dict.update({"total_tokens": getattr(token_usage, "total_tokens")})
                    span.add_event(META_DATA, meta_dict)
        except AttributeError:
            token_usage = None


class LlamaTaskProcessor(TaskProcessor):

    def pre_task_processing(self, to_wrap, wrapped, instance, args, span):
        try:
            if is_root_span(span):
                try:
                    sdk_version = version("monocle_apptrace")
                    span.set_attribute("monocle_apptrace.version", sdk_version)
                except:
                    logger.warning(f"Exception finding monocle-apptrace version.")
            update_spans_with_data_input(to_wrap=to_wrap, wrapped_args=args, span=span)
        except:
            logger.exception("exception in pre_task_processing")

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, return_value, span):
        try:
            update_spans_with_data_output(to_wrap=to_wrap, return_value=return_value, span=span)
            update_span_from_llm_response(response=return_value, span=span, instance=instance)
        except:
            logger.exception("exception in post_task_processing")
