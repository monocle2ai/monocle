"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
import random
import types
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
)


logger = logging.getLogger(__name__)

def patch_instance_method(obj, method_name, func):
    """
    Patch a special method (like __iter__) for a single instance.

    Args:
        obj: the instance to patch
        method_name: the name of the method (e.g., '__iter__')
        func: the new function, expecting (self, ...)
    """
    cls = obj.__class__
    # Dynamically create a new class that inherits from obj's class
    new_cls = type(f"Patched{cls.__name__}", (cls,), {
        method_name: func
    })
    obj.__class__ = new_cls

def extract_messages(kwargs):
    """Extract system and user messages"""
    try:
        messages = []
        if 'instructions' in kwargs:
            messages.append({'instructions': kwargs.get('instructions', {})})
        if 'input' in kwargs:
            messages.append({'input': kwargs.get('input', {})})
        if 'messages' in kwargs and len(kwargs['messages']) >0:
            for msg in kwargs['messages']:
                if msg.get('content') and msg.get('role'):
                    messages.append({msg['role']: msg['content']})

        return [str(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(response, to_wrap=None, span_id=None):
    try:
        if to_wrap and to_wrap.get("stream_prompt_cache") and hasattr(response, '__iter__'):
            original_iter = response.__iter__
            random_id = f"__{random.randint(10**9, 10**10 - 1)}__"
            def new_iter(self):
                stream_prompt_cache = to_wrap.get("stream_prompt_cache")
                if span_id not in stream_prompt_cache:
                    stream_prompt_cache[span_id] = {}
                    stream_prompt_cache[span_id][random_id] = ""
                    
                for item in original_iter():
                    # append to span_id key stream_prompt_cache dict and create the span_id key if it does not exist
                    try:
                        if item.choices and item.choices[0].delta:
                        # append the item to the span_id key
                            stream_prompt_cache[span_id][random_id] += item.choices[0].delta.content
                    except Exception as e:
                        logger.warning("Warning: Error occurred while processing item in new_iter: %s", str(e))
                    finally:
                        yield item

            patch_instance_method(response, '__iter__', new_iter)
            return random_id

        if hasattr(response,"output_text") and len(response.output_text):
            return response.output_text
        if response is not None and hasattr(response,"choices") and len(response.choices) >0:
            if hasattr(response.choices[0],"message"):
                return response.choices[0].message.content
    except (IndexError, AttributeError) as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return None

def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))

def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None


def update_input_span_events(kwargs):
    if 'input' in kwargs and isinstance(kwargs['input'], list):
        query = ' '.join(kwargs['input'])
        return query


def update_output_span_events(results):
    if hasattr(results,'data') and isinstance(results.data, list):
        embeddings = results.data
        embedding_strings = [f"index={e.index}, embedding={e.embedding}" for e in embeddings]
        output = '\n'.join(embedding_strings)
        if len(output) > 100:
            output = output[:100] + "..."
        return output


def update_span_from_llm_response(response):
    meta_dict = {}
    if response is not None and hasattr(response, "usage"):
        if hasattr(response, "usage") and response.usage is not None:
            token_usage = response.usage
        else:
            response_metadata = response.response_metadata
            token_usage = response_metadata.get("token_usage")
        if token_usage is not None:
            meta_dict.update({"completion_tokens": getattr(token_usage,"completion_tokens",None) or getattr(token_usage,"output_tokens",None)})
            meta_dict.update({"prompt_tokens": getattr(token_usage, "prompt_tokens", None) or getattr(token_usage, "input_tokens", None)})
            meta_dict.update({"total_tokens": getattr(token_usage,"total_tokens")})
    return meta_dict

def extract_vector_input(vector_input: dict):
    if 'input' in vector_input:
        return vector_input['input']
    return ""

def extract_vector_output(vector_output):
    try:
        if hasattr(vector_output, 'data') and len(vector_output.data) > 0:
            return vector_output.data[0].embedding
    except Exception as e:
        pass
    return ""

def get_inference_type(instance):
    inference_type: Option[str] = try_option(getattr, instance._client, '_api_version')
    if inference_type.unwrap_or(None):
        return 'azure_openai'
    else:
        return 'openai'