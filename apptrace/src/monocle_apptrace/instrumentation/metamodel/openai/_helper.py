"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import json
import logging
from urllib.parse import urlparse
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_json_dumps,
    try_option,
    get_exception_message,
    get_parent_span,
    get_status_code,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import (
    map_openai_finish_reason_to_finish_type,
    OPENAI_FINISH_REASON_MAPPING
)
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, INFERENCE_AGENT_DELEGATION, INFERENCE_TURN_END, INFERENCE_TOOL_CALL
from contextlib import suppress

logger = logging.getLogger(__name__)

# Mapping of URL substrings to provider names
URL_MAP = {
    "deepseek.com": "deepseek",
    # add more providers here as needed
}

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
                # kwargs['input']
                # [
                #     {
                #         "content": "I need to book a flight from NYC to LAX and also book the Hilton hotel in Los Angeles. Also check the weather in Los Angeles.",
                #         "role": "user"
                #     },
                #     {
                #         "arguments": "{}",
                #         "call_id": "call_dSljcToR2LWwqWibPt0qjeHD",
                #         "name": "transfer_to_flight_agent",
                #         "type": "function_call",
                #         "id": "fc_689c30f96f708191aabb0ffd8098cdbd016ef325124ac05f",
                #         "status": "completed"
                #     },
                #     {
                #         "arguments": "{}",
                #         "call_id": "call_z0MTZroziWDUd0fxVemGM5Pg",
                #         "name": "transfer_to_hotel_agent",
                #         "type": "function_call",
                #         "id": "fc_689c30f99b808191a8743ff407fa8ee2016ef325124ac05f",
                #         "status": "completed"
                #     },
                #     {
                #         "arguments": "{\"city\":\"Los Angeles\"}",
                #         "call_id": "call_rrdRSPv5vcB4pgl6P4W8U2bX",
                #         "name": "get_weather_tool",
                #         "type": "function_call",
                #         "id": "fc_689c30f9b824819196d4ad9379d570f7016ef325124ac05f",
                #         "status": "completed"
                #     },
                #     {
                #         "call_id": "call_rrdRSPv5vcB4pgl6P4W8U2bX",
                #         "output": "The weather in Los Angeles is sunny and 75.",
                #         "type": "function_call_output"
                #     },
                #     {
                #         "call_id": "call_z0MTZroziWDUd0fxVemGM5Pg",
                #         "output": "Multiple handoffs detected, ignoring this one.",
                #         "type": "function_call_output"
                #     },
                #     {
                #         "call_id": "call_dSljcToR2LWwqWibPt0qjeHD",
                #         "output": "{\"assistant\": \"Flight Agent\"}",
                #         "type": "function_call_output"
                #     }
                # ]
                for item in kwargs['input']:
                    if isinstance(item, dict) and 'role' in item and 'content' in item:
                        messages.append({item['role']: item['content']})
                    elif isinstance(item, dict) and 'type' in item and item['type'] == 'function_call':
                        messages.append({
                            "tool_function": item.get("name", ""),
                            "tool_arguments": item.get("arguments", ""),
                            "call_id": item.get("call_id", "")
                        })
                    elif isinstance(item, dict) and 'type' in item and item['type'] == 'function_call_output':
                        messages.append({
                            "call_id": item.get("call_id", ""),
                            "output": item.get("output", "")
                        })
        if 'messages' in kwargs and len(kwargs['messages']) >0:
            for msg in kwargs['messages']:
                if msg.get('content') and msg.get('role'):
                    messages.append({msg['role']: msg['content']})
                elif msg.get('tool_calls') and msg.get('role'):
                    try:
                        tool_call_messages = []
                        for tool_call in msg['tool_calls']:
                            tool_call_messages.append(get_json_dumps({
                                "tool_function": tool_call.function.name,
                                "tool_arguments": tool_call.function.arguments,
                            }))
                        messages.append({msg['role']: tool_call_messages})
                    except Exception as e:
                        logger.warning("Warning: Error occurred while processing tool calls: %s", str(e))

        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(arguments):
    try:
        messages = []
        status = get_status_code(arguments)
        if status == 'success' or status == 'completed':
            response = arguments["result"]
            if hasattr(response, "tools") and isinstance(response.tools, list) and len(response.tools) > 0 and isinstance(response.tools[0], dict):
                tools = []
                for tool in response.tools:
                    tools.append({
                        "tool_id": tool.get("id", ""),
                        "tool_name": tool.get("name", ""),
                        "tool_arguments": tool.get("arguments", "")
                    })
                messages.append({"tools": tools})
            if hasattr(response, "output") and isinstance(response.output, list) and len(response.output) > 0:
                response_messages = []
                role = "assistant"
                for response_message in response.output:
                    if(response_message.type == "function_call"):
                        role = "tools"
                        response_messages.append({
                            "tool_id": response_message.call_id,
                            "tool_name": response_message.name,
                            "tool_arguments": response_message.arguments
                        })
                if len(response_messages) > 0:
                    messages.append({role: response_messages})
                    
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
            return get_json_dumps(messages[0]) if messages else ""
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
    # Try to get host from base_url if it's a parsed object
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    if provider_url.unwrap_or(None) is not None:
        return provider_url.unwrap_or(None)

    # If base_url is just a string (e.g., "https://api.deepseek.com")
    base_url = getattr(instance._client, "base_url", None)
    if isinstance(base_url, str):
        parsed = urlparse(base_url)
        if parsed.hostname:
            return parsed.hostname

    return None


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
            meta_dict.update({"total_tokens": getattr(token_usage,"total_tokens", None)})
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
    # Check if it's Azure OpenAI first
    inference_type: Option[str] = try_option(getattr, instance._client, '_api_version')
    if inference_type.unwrap_or(None):
        return 'azure_openai'

    # Check based on base_url using the mapping
    base_url = getattr(instance, "base_url", None) or getattr(instance._client, "base_url", None)
    
    if base_url:
        base_url_str = str(base_url).lower()
        for key, name in URL_MAP.items():
            if key in base_url_str:
                return name

    # fallback default
    return "openai"

def extract_finish_reason(arguments):
    """Extract finish_reason from OpenAI response"""
    try:
        if "exception" in arguments and arguments["exception"] is not None:
            if hasattr(arguments["exception"], "code") and arguments["exception"].code in OPENAI_FINISH_REASON_MAPPING.keys():
                return arguments["exception"].code
        response = arguments["result"]

        # Handle streaming responses
        if hasattr(response, "finish_reason") and response.finish_reason:
            return response.finish_reason

        # Handle non-streaming responses
        if response is not None and hasattr(response, "choices") and len(response.choices) > 0:
            if hasattr(response.choices[0], "finish_reason"):
                return response.choices[0].finish_reason
    except (IndexError, AttributeError) as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None
    return None

def map_finish_reason_to_finish_type(finish_reason):
    """Map OpenAI finish_reason to finish_type based on the possible errors mapping"""
    return map_openai_finish_reason_to_finish_type(finish_reason)

def agent_inference_type(arguments):
    """Extract agent inference type from OpenAI response"""
    message = json.loads(extract_assistant_message(arguments))
    # message["tools"][0]["tool_name"]
    if message and message.get("tools") and isinstance(message["tools"], list) and len(message["tools"]) > 0:
        agent_prefix = get_value(AGENT_PREFIX_KEY)
        tool_name = message["tools"][0].get("tool_name", "")
        if tool_name and agent_prefix and tool_name.startswith(agent_prefix):
            return INFERENCE_AGENT_DELEGATION
        return INFERENCE_TOOL_CALL
    return INFERENCE_TURN_END

def _get_first_tool_call(response):
    """Helper function to extract the first tool call from various LangChain response formats"""

    with suppress(AttributeError, IndexError, TypeError):
        if response is not None and hasattr(response, "choices") and len(response.choices) > 0:
            if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "tool_calls"):
                tool_calls = response.choices[0].message.tool_calls
                if tool_calls and len(tool_calls) > 0:
                    return tool_calls[0]

    return None

def extract_tool_name(arguments):
    """Extract tool name from OpenAI response when finish_type is tool_call"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"])
        if not tool_call:
            return None

        # Try different name extraction approaches
        for getter in [
            lambda tc: tc.function.name  # object with function.name
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s", str(e))
    
    return None

def extract_tool_type(arguments):
    """Extract tool type from OpenAI response when finish_type is tool_call"""
    try:
        if arguments.get("exception") is not None:
            return None
        
        response = arguments.get("result")
        if response is None:
            return None
        
        # Check for tool calls in the response
        tool_call = _get_first_tool_call(response)
        if tool_call:
            # Return generic tool type for OpenAI tools
            return "tool.openai"
            
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_type: %s", str(e))
        return None
    
    return None


def extract_session_id_from_agents(kwargs):
    # OpenAI Agents passes session via 'session' kwarg
    session = kwargs.get('session')
    if session is not None:
        # Session objects have a session_id attribute
        if hasattr(session, 'session_id'):
            return session.session_id
    return None


