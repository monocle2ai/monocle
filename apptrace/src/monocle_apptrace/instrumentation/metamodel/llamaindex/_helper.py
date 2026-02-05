"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

from ast import arguments
import logging
import time
import threading
from urllib.parse import urlparse
from opentelemetry.sdk.trace import Span
from opentelemetry.context import get_value
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
from monocle_apptrace.instrumentation.metamodel.finish_types import map_llamaindex_finish_reason_to_finish_type
from contextlib import suppress

LLAMAINDEX_AGENT_NAME_KEY = "_active_agent_name"

# Thread-local storage for current agent context
import threading
_thread_local = threading.local()

logger = logging.getLogger(__name__)


def extract_session_id(kwargs):
    # LlamaIndex passes memory via 'memory' kwarg
    memory = kwargs.get('memory')
    if memory is not None:
        if hasattr(memory, 'session_id') and memory.session_id:
            return memory.session_id
        if hasattr(memory, 'chat_store_key') and memory.chat_store_key:
            return memory.chat_store_key
    return None

def get_status(result):
    if result is not None and hasattr(result, 'status'):
        return result.status
    return None

def extract_tools(instance):
    tools = []
    if not hasattr(instance, 'state') or not hasattr(instance.state, 'task_dict'):
        return []
    try:
        data = next(iter(instance.state.task_dict.values())).task
    except (AttributeError, StopIteration):
        return []

    if hasattr(data,'extra_state') and 'sources' in data.extra_state:
        for tool_output in data.extra_state['sources']:
            tool_name = tool_output.tool_name
            if tool_name:
                tools.append(tool_name)
    return tools

def get_tool_type(span):
    if (span.attributes.get("is_mcp", False)):
        return "tool.mcp"
    else:
        return "tool.llamaindex"
    
def get_tool_name(args, instance):
    if len(args) > 1:
        if hasattr(args[1], 'metadata') and hasattr(args[1].metadata, 'name'):
            return args[1].metadata.name
        return ""
    else:
        if hasattr(instance, 'metadata') and hasattr(instance.metadata, 'name'):
            return instance.metadata.name
        return ""

def get_tool_description(arguments):
    if len(arguments['args']) > 1:
        if hasattr(arguments['args'][1], 'metadata') and hasattr(arguments['args'][1].metadata, 'description'):
            return arguments['args'][1].metadata.description
        return ""
    else:
        if hasattr(arguments['instance'], 'metadata') and hasattr(arguments['instance'].metadata, 'description'):
            return arguments['instance'].metadata.description
        return ""

def extract_tool_args(arguments):
    tool_args = {}
    if len(arguments['args']) > 1:
        for key, value in arguments['args'][2].items():
            # check if value is builtin type or a string
            if value is not None and isinstance(value, (str, int, float, bool)):
                tool_args[key] = value
    else:
        for key, value in arguments['kwargs'].items():
            # check if value is builtin type or a string
            if value is not None and isinstance(value, (str, int, float, bool)):
                tool_args[key] = value
    return get_json_dumps(tool_args)

def extract_tool_response(response):
    if hasattr(response, 'raw_output'):
        if hasattr(response.raw_output, 'structuredContent'):
            structured = response.raw_output.structuredContent
            if isinstance(structured, dict) and 'result' in structured:
                return str(structured['result'])
        return response.raw_output
    return ""

def is_delegation_tool(args, instance) -> bool:
    return get_tool_name(args, instance) == "handoff"

def get_agent_name(instance) -> str:
    if hasattr(instance, 'name'):
        return instance.name
    else:
        return instance.__class__.__name__

def get_agent_description(instance) -> str:
    if hasattr(instance, 'description'):
        return instance.description
    return ""

def get_name(instance):
    return instance.name if hasattr(instance, 'name') else ""

def set_current_agent(agent_name: str, agent_span_id: str = None):
    """Store current agent name and span_id in thread-local storage."""
    _thread_local.current_agent = agent_name
    if agent_span_id:
        _thread_local.current_agent_span_id = agent_span_id

def get_current_agent() -> str:
    """Get the current agent name from thread-local storage."""
    return getattr(_thread_local, 'current_agent', '')

def get_current_agent_span_id() -> str:
    """Get current agent span_id from thread-local storage."""
    return getattr(_thread_local, 'current_agent_span_id', '')

def set_from_agent_info(from_agent: str, from_agent_span_id: str):
    """Store from_agent information in thread-local storage for the current agent span."""
    _thread_local.from_agent = from_agent
    _thread_local.from_agent_span_id = from_agent_span_id

def get_from_agent_name() -> str:
    """Get the from_agent name from thread-local storage."""
    return getattr(_thread_local, 'from_agent', None)

def get_from_agent_span_id() -> str:
    """Get the from_agent_span_id from thread-local storage."""
    return getattr(_thread_local, 'from_agent_span_id', None)

# def clear_from_agent_info():
#     """Clear from_agent information from thread-local storage."""
#     if hasattr(_thread_local, 'from_agent'):
#         delattr(_thread_local, 'from_agent')
#     if hasattr(_thread_local, 'from_agent_span_id'):
#         delattr(_thread_local, 'from_agent_span_id')

# Thread-safe store for tracking delegation in concurrent workflows
_delegation_store = {}
_delegation_store_lock = threading.Lock()

def set_delegation_info(target_agent: str, from_agent: str, from_agent_span_id: str):
    """Store delegation information for concurrent workflows."""
    with _delegation_store_lock:
        _delegation_store[target_agent] = {
            'from_agent': from_agent,
            'from_agent_span_id': from_agent_span_id,
            'timestamp': time.time()
        }
    logger.debug(f"set_delegation_info: target={target_agent}, from_agent={from_agent}, span_id={from_agent_span_id}")

def update_delegations_with_span_id(from_agent: str, span_id: str):
    """Update all delegations FROM this agent with the confirmed span_id."""
    with _delegation_store_lock:
        for target_agent, info in _delegation_store.items():
            # Find delegations from this agent that don't have a span_id yet
            if info['from_agent'] == from_agent and (not info['from_agent_span_id'] or info['from_agent_span_id'] == 'N/A'):
                # Check if still recent
                if time.time() - info['timestamp'] < 5.0:
                    info['from_agent_span_id'] = span_id
                    
def get_delegation_info(agent_name: str) -> dict:
    """Retrieve and clear delegation information for an agent."""
    with _delegation_store_lock:
        info = _delegation_store.pop(agent_name, None)
        if info:
            # Check if the delegation info is still recent (within 5 seconds)
            if time.time() - info['timestamp'] < 5.0:
                logger.debug(f"get_delegation_info: agent={agent_name}, found info={info}")
                return info
            else:
                logger.debug(f"get_delegation_info: agent={agent_name}, info expired")
        else:
            logger.debug(f"get_delegation_info: agent={agent_name}, no info found")
        return None

def get_source_agent() -> str:
    """Get the name of the agent that initiated the request."""
    source_agent = get_value(LLAMAINDEX_AGENT_NAME_KEY)
    if source_agent is not None and isinstance(source_agent,str) and source_agent != "":
        return source_agent

    source_agent = get_current_agent()
    if source_agent:
        return source_agent
    return ""

def get_target_agent(results) -> str:
    if hasattr(results, 'raw_input'):
        return results.raw_input.get('kwargs', {}).get("to_agent", "")
    return ""

def extract_messages(args):
    """Extract system and user messages"""
    try:
        messages = []

        def process_message(msg):
            """Processes a single message and extracts relevant information."""
            if hasattr(msg, 'content') and hasattr(msg, 'role'):
                role = getattr(msg.role, 'value', msg.role)
                content = msg.content if role == "system" else extract_query_from_content(msg.content)
                messages.append({role: content})

        if isinstance(args, (list, tuple)) and args:
            for msg in args[0]:
                process_message(msg)
        elif args and isinstance(args, tuple):
            messages.append(args[0])
        if isinstance(args, dict):
            for msg in args.get("messages", []):
                process_message(msg)
        

        return [get_json_dumps(message) for message in messages]

    except Exception as e:
        logger.warning("Error in extract_messages: %s", str(e))
        return []

def extract_agent_input(args):
    if isinstance(args, (list, tuple)):
        input_args = []
        for arg in args:
            if isinstance(arg, (str, dict)):
                input_args.append(arg)
            elif hasattr(arg, 'raw') and isinstance(arg.raw, str):
                input_args.append(arg.raw)
        return input_args
    elif isinstance(args, str):
        return [args]
    return ""

def extract_agent_response(arguments):
    status = get_status_code(arguments)
    if status == 'success':
        if hasattr(arguments['result'], 'response'):
            if hasattr(arguments['result'].response, 'content'):
                return arguments['result'].response.content
            return arguments['result'].response
        return ""
    else:
        if arguments["exception"] is not None:
            return get_exception_message(arguments)
        elif hasattr(arguments['result'], "error"):
            return arguments['result'].error

def extract_assistant_message(arguments):
    status = get_status_code(arguments)
    messages = []
    role = "assistant"
    if status == 'success':
        if isinstance(arguments['result'], str):
            messages.append({role: arguments['result']})
        if hasattr(arguments['result'], "content"):
            messages.append({role: arguments['result'].content})
        if hasattr(arguments['result'], "message") and hasattr(arguments['result'].message, "content"):
            role = getattr(arguments['result'].message, 'role', role)
            if hasattr(role, 'value'):
                role = role.value
            messages.append({role: arguments['result'].message.content})
        if hasattr(arguments['result'],"response") and isinstance(arguments['result'].response, str):
            messages.append({role: arguments['result'].response})
    else:
        if arguments["exception"] is not None:
            return get_exception_message(arguments)
        elif hasattr(arguments['result'], "error"):
            return arguments['result'].error

    return get_json_dumps(messages[0]) if messages else ""

def extract_query_from_content(content):
    try:
        query_prefix = "Query:"
        answer_prefix = "Answer:"
        query_start = content.find(query_prefix)
        if query_start == -1:
            return content

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
    if hasattr(instance,'api_base'):
        provider_url: Option[str]= try_option(getattr, instance, 'api_base').and_then(lambda url: urlparse(url).hostname)
    if hasattr(instance,'_client'):
        provider_url:Option[str] = try_option(getattr, instance._client.base_url,'host')
    if hasattr(instance, 'model') and isinstance(instance.model, str) and 'gemini' in instance.model.lower():
        provider_url: Option[str] = try_option(lambda: 'gemini.googleapis.com')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    if hasattr(instance,'_client'):
        if hasattr(instance._client,'sdk_configuration'):
            inference_endpoint: Option[str] = try_option(getattr, instance._client.sdk_configuration, 'server_url').map(str)
        if hasattr(instance._client,'base_url'):
            inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if hasattr(instance, 'model') and isinstance(instance.model, str) and 'gemini' in instance.model.lower():
        inference_endpoint = try_option(lambda: f"https://generativelanguage.googleapis.com/v1beta/models/{instance.model}:generateContent")
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


def update_input_span_events(args):
    if isinstance(args, tuple):
        return args[0].query_str if len(args) > 0 else ""


def update_output_span_events(results):
    if isinstance(results, list) and len(results) >0:
        output_arg_text = results[0].text
        if len(output_arg_text) > 100:
            output_arg_text = output_arg_text[:100] + "..."
        return output_arg_text


def update_span_from_llm_response(response, instance):
    meta_dict = {}
    if response is not None and hasattr(response, "raw"):
        if response.raw is not None:
            token_usage = response.raw.get("usage") if isinstance(response.raw, dict) else getattr(response.raw, "usage", None)
            if token_usage is None:
                token_usage = response.raw.get("usage_metadata") if isinstance(response.raw, dict) else getattr(response.raw,
                                                                                                       "usage_metadata", None)
            if token_usage is not None:
                temperature = instance.__dict__.get("temperature", None)
                meta_dict.update({"temperature": temperature})
                meta_dict.update({"completion_tokens": getattr(token_usage, "completion_tokens",None) or getattr(token_usage,"output_tokens",None) or token_usage.get("candidates_token_count",None)})
                meta_dict.update({"prompt_tokens": getattr(token_usage, "prompt_tokens",None) or getattr(token_usage,"input_tokens",None) or token_usage.get("prompt_token_count",None)})
                total_tokens = getattr(token_usage, "total_tokens", None)
                if total_tokens is not None:
                    meta_dict.update({"total_tokens": total_tokens})
                else:
                    output_tokens = getattr(token_usage, "output_tokens", None)
                    input_tokens = getattr(token_usage, "input_tokens", None)
                    if output_tokens is not None and input_tokens is not None:
                        meta_dict.update({"total_tokens": output_tokens + input_tokens})
                    else:
                        meta_dict.update({ "total_tokens": token_usage.get("total_token_count", None)})

    return meta_dict

def extract_finish_reason(arguments):
    """Extract finish_reason from LlamaIndex response."""
    try:
        # Handle exception cases first
        if arguments.get("exception") is not None:
            return "error"
        
        response = arguments.get("result")
        if response is None:
            return None
            
        # Check various possible locations for finish_reason in LlamaIndex responses
        
        # Direct finish_reason attribute
        if hasattr(response, "finish_reason") and response.finish_reason:
            return response.finish_reason
            
        # Check if response has raw attribute (common in LlamaIndex)
        if hasattr(response, "raw") and response.raw:
            raw_response = response.raw
            if isinstance(raw_response, dict):
                # Check for finish_reason in raw response
                if "finish_reason" in raw_response:
                    return raw_response["finish_reason"]
                if "stop_reason" in raw_response:
                    return raw_response["stop_reason"]
                # Check for choices structure (OpenAI-style)
                if "choices" in raw_response and raw_response["choices"]:
                    choice = raw_response["choices"][0]
                    if isinstance(choice, dict) and "finish_reason" in choice:
                        return choice["finish_reason"]
            elif hasattr(raw_response, "choices") and raw_response.choices:
                # Handle object-style raw response
                choice = raw_response.choices[0]
                if hasattr(choice, "finish_reason"):
                    return choice.finish_reason
        
        # Check for additional metadata
        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            kwargs = response.additional_kwargs
            if isinstance(kwargs, dict):
                for key in ["finish_reason", "stop_reason"]:
                    if key in kwargs:
                        return kwargs[key]
        
        # Check for response metadata
        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if isinstance(metadata, dict):
                for key in ["finish_reason", "stop_reason"]:
                    if key in metadata:
                        return metadata[key]
        
        # Check for source nodes or other LlamaIndex-specific attributes
        if hasattr(response, "source_nodes") and response.source_nodes:
            # If we have source nodes, it's likely a successful retrieval
            return "stop"
        
        # If no specific finish reason found, infer from status
        status_code = get_status_code(arguments)
        if status_code == 'success':
            return "stop"  # Default success finish reason
        elif status_code == 'error':
            return "error"
            
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None
    
    return None


def map_finish_reason_to_finish_type(finish_reason):
    """Map LlamaIndex finish_reason to finish_type."""
    return map_llamaindex_finish_reason_to_finish_type(finish_reason)

def extract_agent_request_input(kwargs):
    if "user_msg" in kwargs:
        return kwargs["user_msg"]
    return ""

def extract_agent_request_output(arguments):
    if hasattr(arguments['result'], 'response'):
        if hasattr(arguments['result'].response, 'content'):
            return arguments['result'].response.content
        return arguments['result'].response
    elif hasattr(arguments['result'], 'raw_output'):
        return arguments['result'].raw_output
    return ""

def _get_first_tool_call(response):
    """Helper function to extract the first tool call from various LangChain response formats"""

    with suppress(AttributeError, IndexError, TypeError):
        if hasattr(response, "raw") and response.raw:
            raw_response = response.raw

            if hasattr(raw_response, "choices") and raw_response.choices:
                choice = raw_response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "tool_calls"):
                    tool_calls = choice.message.tool_calls
                    if tool_calls and len(tool_calls) > 0:
                        return tool_calls[0]

    return None

def extract_tool_name(arguments):
    """Extract tool name from LlamaIndex response when finish_type is tool_call"""
    try:
        finish_reason = extract_finish_reason(arguments)
        finish_type = map_finish_reason_to_finish_type(finish_reason)
        
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"])
        if not tool_call:
            return None

        for getter in [
            lambda tc: tc['function']['name'],
            lambda tc: tc.function.name
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s", str(e))
    
    return None

def extract_tool_type(arguments):
    """Extract tool type from LlamaIndex response when finish_type is tool_call"""
    try:
        finish_reason = extract_finish_reason(arguments)
        finish_type = map_finish_reason_to_finish_type(finish_reason)
        
        if finish_type != "tool_call":
            return None

        tool_name = extract_tool_name(arguments)
        if tool_name:
            return TOOL_TYPE
            
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_type: %s", str(e))
    
    return None
