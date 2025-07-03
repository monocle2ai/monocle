"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    try_option,
    get_exception_message,
    get_parent_span,
    get_status_code,
)
from monocle_apptrace.instrumentation.common.span_handler import NonFrameworkSpanHandler, WORKFLOW_TYPE_MAP

logger = logging.getLogger(__name__)


def extract_messages(kwargs):
    """Extract system and user messages"""
    try:
        messages = []
        if 'instructions' in kwargs:
            messages.append({'system': kwargs.get('instructions', {})})
        if 'input' in kwargs:
            if isinstance(kwargs['input'], str): 
                messages.append({'user': kwargs.get('input', "")})
            # [
            #     {
            #         "role": "developer",
            #         "content": "Talk like a pirate."
            #     },
            #     {
            #         "role": "user",
            #         "content": "Are semicolons optional in JavaScript?"
            #     }
            # ]
            if isinstance(kwargs['input'], list):
                for item in kwargs['input']:
                    if isinstance(item, dict) and 'role' in item and 'content' in item:
                        messages.append({item['role']: item['content']})
        if 'messages' in kwargs and len(kwargs['messages']) >0:
            for msg in kwargs['messages']:
                if msg.get('content') and msg.get('role'):
                    messages.append({msg['role']: msg['content']})

        return [str(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(arguments):
    try:
        messages = []
        status = get_status_code(arguments)
        if status == 'success' or status == 'completed':
            response = arguments["result"]
            if hasattr(response, "output_text") and len(response.output_text):
                role = response.role if hasattr(response, "role") else "assistant"
                messages.append({role: response.output_text})
            if (
                response is not None
                and hasattr(response, "choices")
                and len(response.choices) > 0
            ):
                if hasattr(response.choices[0], "message"):
                    role = (
                        response.choices[0].message.role
                        if hasattr(response.choices[0].message, "role")
                        else "assistant"
                    )
                    messages.append({role: response.choices[0].message.content})
            return [str(message) for message in messages][0] if messages else ""
        else:
            if arguments["exception"] is not None:
                return get_exception_message(arguments)
            elif hasattr(arguments["result"], "error"):
                return arguments["result"].error
        
    except (IndexError, AttributeError) as e:
        logger.warning(
            "Warning: Error occurred in extract_assistant_message: %s", str(e)
        )
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

class OpenAISpanHandler(NonFrameworkSpanHandler):
    def is_teams_span_in_progress(self) -> bool:
        return self.is_framework_span_in_progess() and self.get_workflow_name_in_progress() == WORKFLOW_TYPE_MAP["teams.ai"]

    # If openAI is being called by Teams AI SDK, then retain the metadata part of the span events
    def skip_processor(self, to_wrap, wrapped, instance, span, args, kwargs) -> list[str]:
        if self.is_teams_span_in_progress():
            return ["attributes", "events.data.input", "events.data.output"]
        else:
            return super().skip_processor(to_wrap, wrapped, instance, span, args, kwargs)

    def hydrate_events(self, to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=None, ex:Exception=None) -> bool:
        # If openAI is being called by Teams AI SDK, then copy parent
        if self.is_teams_span_in_progress() and ex is None:
            return super().hydrate_events(to_wrap, wrapped, instance, args, kwargs, ret_result, span=parent_span, parent_span=None, ex=ex)

        return super().hydrate_events(to_wrap, wrapped, instance, args, kwargs, ret_result, span, parent_span=parent_span, ex=ex)
