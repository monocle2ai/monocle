"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
import json
from io import BytesIO
from functools import wraps

from rfc3986 import urlparse
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, INFERENCE_AGENT_DELEGATION, INFERENCE_TOOL_CALL, INFERENCE_TURN_END, TOOL_TYPE
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import ( get_exception_message, get_json_dumps, get_status_code,)
from monocle_apptrace.instrumentation.metamodel.finish_types import map_bedrock_finish_reason_to_finish_type
from contextlib import suppress

logger = logging.getLogger(__name__)


def extract_messages(args):
    """Extract system and user messages"""
    try:
        messages = []
        if args and isinstance(args, dict) and len(args) > 0:
            if 'Body' in args and isinstance(args['Body'], str):
                data = json.loads(args['Body'])
                question = data.get("question")
                messages.append(question)
            if 'messages' in args and isinstance(args['messages'], list):
                role = args['messages'][0]['role']
                user_message = extract_query_from_content(args['messages'][0]['content'][0]['text'])
                messages.append({role: user_message})
        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []

def get_exception_status_code(arguments):
    if arguments['exception'] is not None and hasattr(arguments['exception'], 'response') and arguments['exception'].response is not None:
        if "ResponseMetadata" in arguments['exception'].response and "HTTPStatusCode" in arguments['exception'].response["ResponseMetadata"]:
            return arguments['exception'].response["ResponseMetadata"]["HTTPStatusCode"]
    elif arguments['exception'] is not None:
        return 'error'
    else:
        return 'success'

def extract_assistant_message(arguments):
    try:
        status = get_status_code(arguments)
        messages = []
        role = "assistant"
        if status == 'success':
            if "Body" in arguments['result'] and hasattr(arguments['result']['Body'], "_raw_stream"):
                raw_stream = getattr(arguments['result']['Body'], "_raw_stream")
                if hasattr(raw_stream, "data"):
                    response_bytes = getattr(raw_stream, "data")
                    response_str = response_bytes.decode('utf-8')
                    response_dict = json.loads(response_str)
                    arguments['result']['Body'] = BytesIO(response_bytes)
                    messages.append({role: response_dict["answer"]})
            if "output" in arguments['result']:
                output = arguments['result'].get("output", {})
                message = output.get("message", {})
                content = message.get("content", [])
                if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                    reply = content[0]["text"]
                    messages.append({role: reply})
                else:
                    tool_call = _get_first_tool_call(arguments['result'])
                    if tool_call is not None:
                        messages.append({role: str(tool_call['toolUse']['input'])})
        else:
            if arguments["exception"] is not None:
                return get_exception_message(arguments)
            elif hasattr(arguments["result"], "error"):
                return arguments["result"].error
        return get_json_dumps(messages[0]) if messages else ""
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return []


def extract_query_from_content(content:str) -> str:
    try:
        query_prefix = "Query:"
        answer_prefix = "Answer:"
        query_start = content.find(query_prefix)
        if query_start != -1:
            query_start += len(query_prefix)
        else:
            query_start = None
        answer_start = content.find(answer_prefix, query_start)
        if answer_start == -1:
            query = content[query_start:].strip()
        else:
            query = content[query_start:answer_start].strip()
        return query
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_query_from_content: %s", str(e))
        return ""


def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None

def update_span_from_llm_response(response, instance):
    meta_dict = {}
    if response is not None and isinstance(response, dict) and "usage" in response:
        token_usage = response["usage"]

        if token_usage is not None:
            temperature = instance.__dict__.get("temperature", None)
            meta_dict.update({"temperature": temperature})
            meta_dict.update({"completion_tokens": resolve_from_alias(token_usage,["completion_tokens","output_tokens","outputTokens"])})
            meta_dict.update({"prompt_tokens": resolve_from_alias(token_usage,["prompt_tokens","input_tokens","inputTokens"])})
            meta_dict.update({"total_tokens": resolve_from_alias(token_usage,["total_tokens","totalTokens"])})
    return meta_dict


def extract_finish_reason(arguments):
    """Extract finish_reason/stopReason from Bedrock response."""
    try:
        # Handle exception cases first
        if arguments.get("exception") is not None:
            return "error"
        
        result = arguments.get("result")
        if result is None:
            return None
            
        # Check various possible locations for finish_reason in Bedrock responses
        
        # Direct stopReason attribute (Bedrock Converse API)
        if "stopReason" in result:
            return result["stopReason"]
            
        # Check for completionReason (some Bedrock models)
        if "completionReason" in result:
            return result["completionReason"]
            
        # Check for output structure (Bedrock Converse API)
        if "output" in result and "message" in result["output"]:
            message = result["output"]["message"]
            if "stopReason" in message:
                return message["stopReason"]
                
        # Check for nested result structure
        if "result" in result:
            nested_result = result["result"]
            if "stopReason" in nested_result:
                return nested_result["stopReason"]
            if "completionReason" in nested_result:
                return nested_result["completionReason"]
                
        # Check for streaming response accumulated finish reason
        if "type" in result and result["type"] == "stream":
            if "stopReason" in result:
                return result["stopReason"]
                
        # Check for response metadata
        if "ResponseMetadata" in result:
            metadata = result["ResponseMetadata"]
            if "stopReason" in metadata:
                return metadata["stopReason"]
                
        # Check for Body content (for some Bedrock responses)
        if "Body" in result:
            body = result["Body"]
            if hasattr(body, "_raw_stream"):
                raw_stream = getattr(body, "_raw_stream")
                if hasattr(raw_stream, "data"):
                    response_bytes = getattr(raw_stream, "data")
                    response_str = response_bytes.decode('utf-8')
                    try:
                        response_dict = json.loads(response_str)
                        if "stopReason" in response_dict:
                            return response_dict["stopReason"]
                        if "completionReason" in response_dict:
                            return response_dict["completionReason"]
                    except json.JSONDecodeError:
                        pass
                        
        # If no specific finish reason found, infer from status
        status_code = get_status_code(arguments)
        if status_code == 'success':
            return "end_turn"  # Default successful completion
        elif status_code == 'error':
            return "error"
            
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_finish_reason: %s", str(e))
        return None
    
    return None


def map_finish_reason_to_finish_type(finish_reason):
    """Map Bedrock finish_reason/stopReason to finish_type."""
    return map_bedrock_finish_reason_to_finish_type(finish_reason)

def extract_provider_name(instance):
    return urlparse(instance.meta.endpoint_url).hostname

def _get_first_tool_call(response):
    """Helper function to extract the first tool call from various Boto response formats"""
    with suppress(AttributeError, IndexError, TypeError):
        if "output" in response and "message" in response["output"]:
            message = response["output"]["message"]
            if "content" in message and isinstance(message["content"], list):
                for content_block in reversed(message["content"]):
                    if "toolUse" in content_block:
                        return content_block

    return None

def extract_tool_name(arguments):
    """Extract tool name from Bedrock response when finish_type is tool_call"""
    try:
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        if finish_type != "tool_call":
            return None

        tool_call = _get_first_tool_call(arguments["result"])
        if not tool_call:
            return None

        for getter in [
            lambda tc: tc["toolUse"]["name"],  # dict with name key
        ]:
            try:
                return getter(tool_call)
            except (KeyError, AttributeError, TypeError):
                continue

    except Exception as e:
        logger.warning("Warning: Error occurred in extract_tool_name: %s", str(e))

    return None

def extract_tool_type(arguments):
    """Extract tool type from Bedrock response when finish_type is tool_call"""
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

def agent_inference_type(arguments):
    """Extract agent inference type from Bedrock response"""
    try:
        # Check finish_type to determine the inference type
        finish_type = map_finish_reason_to_finish_type(extract_finish_reason(arguments))
        
        if finish_type == "tool_call":
            tool_call = _get_first_tool_call(arguments["result"])
            if tool_call:
                tool_name = tool_call.get("toolUse", {}).get("name", "")
                agent_prefix = get_value(AGENT_PREFIX_KEY)
                if agent_prefix and tool_name.startswith(agent_prefix):
                    return INFERENCE_AGENT_DELEGATION
            return INFERENCE_TOOL_CALL
        
        return INFERENCE_TURN_END
    except Exception as e:
        logger.warning("Warning: Error occurred in agent_inference_type: %s", str(e))
        return INFERENCE_TURN_END
