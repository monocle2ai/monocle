"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import json
import logging
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    get_keys_as_tuple,
    get_nested_value,
    get_status_code,
    try_option,
    get_exception_message,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import map_mistral_finish_reason_to_finish_type
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, INFERENCE_AGENT_DELEGATION, INFERENCE_TURN_END, INFERENCE_TOOL_CALL, TOOL_TYPE
from contextlib import suppress

logger = logging.getLogger(__name__)


def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)

def update_input_span_events(kwargs):
    """Extract embedding input for spans"""
    if "inputs" in kwargs and isinstance(kwargs["inputs"], list):
        # Join multiple strings into one
        return " | ".join(kwargs["inputs"])
    elif "inputs" in kwargs and isinstance(kwargs["inputs"], str):
        return kwargs["inputs"]
    return ""

def update_output_span_events(results):
    """Extract embedding output for spans"""
    try:
        if hasattr(results, "data") and isinstance(results.data, list):
            embeddings = results.data
            # just return the indices, not full vectors
            embedding_summaries = [
                f"index={e.index}, dim={len(e.embedding)}"
                for e in embeddings
            ]
            output = "\n".join(embedding_summaries)
            if len(output) > 200:
                output = output[:200] + "..."
            return output
    except Exception as e:
        logger.warning("Error in update_output_span_events: %s", str(e))
    return ""

def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))


def dummy_method(arguments):
    pass


def extract_messages(kwargs):
    """Extract system and user messages"""
    try:
        messages = []
        if "system" in kwargs and isinstance(kwargs["system"], str):
            messages.append({"system": kwargs["system"]})
        if 'messages' in kwargs and kwargs['messages']:
            for msg in kwargs['messages']:
                if msg.get('content') and msg.get('role'):
                    messages.append({msg['role']: msg['content']})
        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def get_exception_status_code(arguments):
    exc = arguments.get("exception")
    if exc is not None and hasattr(exc, "status_code"):
        if exc.status_code == 401:
            return "unauthorized"
        elif exc.status_code == 403:
            return "forbidden"
        elif exc.status_code == 404:
            return "not_found"
        else:
            return str(exc.status_code)
    elif exc is not None:
        return "error"
    else:
        return "success"


def extract_assistant_message(arguments):
    """
    Extract the assistant message from a Mistral response or stream chunks.
    Returns a JSON string like {"assistant": "<text>"}.
    """
    try:
        result = arguments.get("result") if isinstance(arguments, dict) else arguments
        if result is None:
            return ""

        # Handle full response
        if hasattr(result, "choices") and result.choices:
            msg_obj = result.choices[0].message
            return get_json_dumps({msg_obj.role: msg_obj.content})

        # Handle streaming: result might be a list of CompletionEvent chunks
        if isinstance(result, list):
            content = []
            for chunk in result:
                if hasattr(chunk, "data") and hasattr(chunk.data, "choices") and chunk.data.choices:
                    choice = chunk.data.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        content.append(choice.delta.content or "")
            return get_json_dumps({"assistant": "".join(content)})

        return ""

    except Exception as e:
        logger.warning("Warning in extract_assistant_message: %s", str(e))
        return ""


'''def update_span_from_llm_response(response):
    meta_dict = {}
    if response is not None and hasattr(response, "usage"):
        token_usage = getattr(response, "usage", None) or getattr(response, "response_metadata", {}).get("token_usage")
        if token_usage is not None:
            meta_dict.update({"completion_tokens": getattr(response.usage, "output_tokens", 0)})
            meta_dict.update({"prompt_tokens": getattr(response.usage, "input_tokens", 0)})
            meta_dict.update({"total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)})
    return meta_dict'''

def update_span_from_llm_response(result, include_token_counts=False):
    tokens = {}
    if include_token_counts and result is not None:
        # Try to extract token usage from the response
        if hasattr(result, "usage"):
            usage = result.usage
            tokens = {
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

    return tokens


def extract_finish_reason(arguments):
    """
    Extract finish_reason from a Mistral response or stream chunks.
    Works for both streaming (list of chunks) and full responses.
    """
    try:
        response = arguments.get("result") if isinstance(arguments, dict) else arguments
        if response is None:
            return None

        # Handle full response: check choices for finish_reason
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason:
                return finish_reason

        # Handle streaming: list of chunks, last chunk may have finish_reason
        if isinstance(response, list):
            for chunk in reversed(response):
                if hasattr(chunk, "data") and hasattr(chunk.data, "choices") and chunk.data.choices:
                    fr = getattr(chunk.data.choices[0], "finish_reason", None)
                    if fr is not None:
                        return fr

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None

    return None


def map_finish_reason_to_finish_type(finish_reason):
    """Map Mistral stop_reason to finish_type, similar to OpenAI mapping."""
    return map_mistral_finish_reason_to_finish_type(finish_reason)


def agent_inference_type(arguments):
    """Extract agent inference type from Mistral response"""
    try:
        status = get_status_code(arguments)
        if status in ('success', 'completed'):
            response = arguments.get("result")
            if response is None:
                return INFERENCE_TURN_END

            # Check if finish_reason indicates tool use (Mistral uses "tool_calls")
            if hasattr(response, "choices") and response.choices:
                finish_reason = getattr(response.choices[0], "finish_reason", None)

                if finish_reason == "tool_calls":
                    # Check if we have tool calls in the message
                    if (hasattr(response.choices[0], "message") and
                        hasattr(response.choices[0].message, "tool_calls") and
                        response.choices[0].message.tool_calls):

                        tool_calls = response.choices[0].message.tool_calls

                        agent_prefix = get_value(AGENT_PREFIX_KEY)
                        for tool_call in tool_calls:
                            if hasattr(tool_call, "function") and hasattr(tool_call.function, "name"):
                                tool_name = tool_call.function.name
                                if agent_prefix and tool_name.startswith(agent_prefix):
                                    return INFERENCE_AGENT_DELEGATION

                        return INFERENCE_TOOL_CALL

            # Fallback: check the extracted message for tool content
            assistant_message = extract_assistant_message(arguments)
            if assistant_message:
                try:
                    message = json.loads(assistant_message)
                    assistant_content = message.get("assistant", "") if isinstance(message, dict) else ""
                    agent_prefix = get_value(AGENT_PREFIX_KEY)
                    if agent_prefix and agent_prefix in assistant_content:
                        return INFERENCE_AGENT_DELEGATION
                except (json.JSONDecodeError, TypeError):
                    agent_prefix = get_value(AGENT_PREFIX_KEY)
                    if agent_prefix and agent_prefix in assistant_message:
                        return INFERENCE_AGENT_DELEGATION

        return INFERENCE_TURN_END

    except Exception as e:
        logger.warning("Warning: Error occurred in agent_inference_type: %s", str(e))
        return INFERENCE_TURN_END

def _get_first_tool_call(response):

    with suppress(AttributeError, IndexError, TypeError):
        if response is not None and hasattr(response, "choices") and len(response.choices) > 0:
            if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "tool_calls"):
                tool_calls = response.choices[0].message.tool_calls
                if tool_calls and len(tool_calls) > 0:
                    return tool_calls[0]

    return None

def extract_tool_name(arguments):
    """Extract tool name from Mistral response when finish_type is tool_call"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"])
        if not tool_call:
            return None

        for getter in [
            lambda tc: tc.function.name,
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s", str(e))
    
    return None

def extract_tool_type(arguments):
    """Extract tool type from Mistral response when finish_type is tool_call"""
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
