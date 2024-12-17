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
        if args and isinstance(args, tuple) and len(args) > 0:
            if hasattr(args[0], "messages") and isinstance(args[0].messages, list):
                for msg in args[0].messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        messages.append({msg.type: msg.content})
        return [str(d) for d in messages]
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


def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance.client._client.base_url, 'host')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance.client._client, 'base_url').map(str)
    if inference_endpoint.is_none():
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

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


def update_span_with_context_input(to_wrap, wrapped_args, span: Span):
    package_name: str = to_wrap.get('package')
    if "retrievers" in package_name and len(wrapped_args) > 0:
        input_arg_text = wrapped_args[0]
        if input_arg_text:
            span.add_event(DATA_INPUT_KEY, {QUERY: input_arg_text})


def update_span_with_context_output(to_wrap, return_value, span: Span):
    package_name: str = to_wrap.get('package')
    if "retrievers" in package_name:
        output_arg_text = " ".join([doc.page_content for doc in return_value if hasattr(doc, 'page_content')])
        if len(output_arg_text) > 100:
            output_arg_text = output_arg_text[:100] + "..."
        span.add_event(DATA_OUTPUT_KEY, {RESPONSE: output_arg_text})


def update_span_with_prompt_input(to_wrap, wrapped_args, span: Span):
    input_arg_text = wrapped_args[0]
    if isinstance(input_arg_text, dict):
        span.add_event(PROMPT_INPUT_KEY, input_arg_text)
    else:
        span.add_event(PROMPT_INPUT_KEY, {QUERY: input_arg_text})


def update_span_with_prompt_output(to_wrap, wrapped_args, span: Span):
    if isinstance(wrapped_args, str):
        span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: wrapped_args})
    elif isinstance(wrapped_args, dict):
        span.add_event(PROMPT_OUTPUT_KEY, wrapped_args)

def update_span_from_llm_response(response, span: Span, instance):
    if response is not None and hasattr(response, "response_metadata"):
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            token_usage = response.usage_metadata
        else:
            response_metadata = response.response_metadata
            token_usage = response_metadata.get("token_usage")

        meta_dict = {}
        if token_usage is not None:
            temperature = instance.__dict__.get("temperature", None)
            meta_dict.update({"temperature": temperature})
            meta_dict.update(
                {"completion_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens")})
            meta_dict.update({"prompt_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens")})
            meta_dict.update({"total_tokens": token_usage.get("total_tokens")})
            span.add_event(META_DATA, meta_dict)


class LangchainTaskProcessor(TaskProcessor):

    def pre_task_processing(self, to_wrap, wrapped, instance, args, span):
            try:
                if is_root_span(span):
                    try:
                        sdk_version = version("monocle_apptrace")
                        span.set_attribute("monocle_apptrace.version", sdk_version)
                    except:
                        logger.warning(f"Exception finding monocle-apptrace version.")
                    update_span_with_prompt_input(to_wrap=to_wrap, wrapped_args=args, span=span)
                update_span_with_context_input(to_wrap=to_wrap, wrapped_args=args, span=span)
            except:
                logger.exception("exception in pre_task_processing")

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, return_value, span):
        try:
            update_span_with_context_output(to_wrap=to_wrap, return_value=return_value, span=span)
            update_span_from_llm_response(response=return_value, span=span, instance=instance)

            if is_root_span(span):
                update_span_with_prompt_output(to_wrap=to_wrap, wrapped_args=return_value, span=span)
        except:
            logger.exception("exception in post_task_processing")


