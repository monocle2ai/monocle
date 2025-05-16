"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
)
from opentelemetry.trace.status import StatusCode
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.constants import WORKFLOW_TYPE_KEY
from opentelemetry.context import get_current
from opentelemetry.trace import Span
from opentelemetry.trace.propagation import _SPAN_KEY
WORKFLOW_TYPE_MAP = {
    "llama_index": "workflow.llamaindex",
    "langchain": "workflow.langchain",
    "haystack": "workflow.haystack",
    "teams.ai": "workflow.teams_ai",
}

logger = logging.getLogger(__name__)


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


def extract_assistant_message(response):
    try:
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

def hydrate_event_metadata(to_wrap, wrapped, instance, args, kwargs, result, span):
    output_processor = to_wrap.get("output_processor")
    if not output_processor:
        return

    events = output_processor.get("events", [])
    context = {"instance": instance, "args": args, "kwargs": kwargs, "result": result}

    for event in events:
        if event.get("name") != "metadata":
            continue
        event_attributes= {}
        for attribute in event.get("attributes", []):
            attribute_key = attribute.get("attribute")
            accessor = attribute.get("accessor")
            if not accessor:
                continue
            try:
                accessed_value = accessor(context)
                if isinstance(accessed_value, dict):
                    accessed_value = {k: v for k, v in accessed_value.items() if v is not None}
                if accessed_value is not None and isinstance(accessed_value, (str, list, dict)):
                    if attribute_key:
                        event_attributes[attribute_key] = accessed_value
                    else:
                        if isinstance(accessed_value, dict):
                            event_attributes.update(accessed_value)
            except MonocleSpanException as e:
                span.set_status(StatusCode.ERROR, str(e))
            except Exception as e:
                logger.debug(f"[hydrate_event_metadata] Failed to evaluate accessor for '{attribute_key}': {e}")

        span.add_event(name="metadata", attributes=event_attributes)



class OpenAIFrameworkSpanHandler(SpanHandler):

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        try:
            # pdb.set_trace()
            _parent_span_context = get_current()
            if _parent_span_context is not None:
                parent_span: Span = _parent_span_context.get(_SPAN_KEY, None)
                if parent_span is not None and parent_span.name.startswith("teams") == True:
                    hydrate_event_metadata(to_wrap, wrapped, instance, args, kwargs, return_value, parent_span)
        except Exception as e:
            logger.info(f"Failed to propogate flask response: {e}")
        super().post_tracing(to_wrap, wrapped, instance, args, kwargs, return_value)

    def skip_processor(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        return get_value(WORKFLOW_TYPE_KEY) in WORKFLOW_TYPE_MAP.values()