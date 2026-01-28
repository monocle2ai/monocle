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
            # Check if this is invoke_data_automation_async
            if 'inputConfiguration' in args:
                return extract_data_automation_input(args)
            # Check if this is invoke_model
            elif 'body' in args:
                return extract_invoke_model_messages(args)
            # Original converse API handling
            elif 'Body' in args and isinstance(args['Body'], str):
                data = json.loads(args['Body'])
                question = data.get("question")
                messages.append(question)
            elif 'messages' in args and isinstance(args['messages'], list):
                role = args['messages'][0]['role']
                user_message = extract_query_from_content(args['messages'][0]['content'][0]['text'])
                messages.append({role: user_message})
            elif 'input' in args and 'text' in args['input']:
                messages.append(args['input']['text'])
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
                if 'text' in output:
                    messages.append(output['text'])
                if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                    reply = content[0]["text"]
                    messages.append({role: reply})
                else:
                    tool_call = _get_first_tool_call(arguments['result'])
                    if tool_call is not None:
                        messages.append({role: str(tool_call['toolUse']['input'])})

            if 'invocationArn' in arguments['result']:
                result= arguments['result']
                response_info = {
                    'invocation_arn': result.get('invocationArn', ''),
                }
                messages.append({'data_automation_response': response_info})
            # Check if this is invoke_model by looking at result structure
            if 'body' in arguments['result']:
                result_body = arguments['result'].get('body')
                if hasattr(result_body, 'read'):
                    output = extract_invoke_model_response(arguments['result'])
                    messages.append(output)
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

def get_model(kwargs):
    if 'retrieveAndGenerateConfiguration' in kwargs and 'knowledgeBaseConfiguration' in kwargs['retrieveAndGenerateConfiguration']:
        kb = kwargs['retrieveAndGenerateConfiguration']['knowledgeBaseConfiguration']
        if 'modelArn' in kb:
            return kb['modelArn'].split('/')[1]
    return ''

def get_vector_db(kwargs):
    if 'retrieveAndGenerateConfiguration' in kwargs and 'type' in kwargs['retrieveAndGenerateConfiguration']:
        return kwargs['retrieveAndGenerateConfiguration']['type'].lower()
    return ''

def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None

def update_span_from_llm_response(response, instance):
    meta_dict = {}
    token_usage =None
    if response is not None and isinstance(response, dict) and "usage" in response:
        token_usage = response["usage"]
    if response is not None and isinstance(response, dict) and 'body' in response:
        token_usage = update_span_from_invoke_model_response(response, instance)
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

def extract_retrieval_query(kwargs):
    """Extract the query from retrieve_and_generate or retrieve input"""
    try:
        messages = []
        # Check for retrieve_and_generate format
        if 'input' in kwargs:
            input_data = kwargs['input']
            if 'text' in input_data:
                messages.append(input_data['text'])

        return [get_json_dumps(message) for message in messages] if messages else []
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_retrieval_query: %s", str(e))
        return []


def extract_retrieval_response(arguments):
    """Extract response from retrieve_and_generate or retrieve result"""
    try:
        status = get_status_code(arguments)
        messages = []
        
        if status == 'success' and 'result' in arguments:
            result = arguments['result']
            response_data = {}
            # Extract retrieved references from citations
            if 'citations' in result and result['citations']:
                retrieved_refs = []
                for citation in result['citations']:
                    if 'retrievedReferences' in citation:
                        for ref in citation['retrievedReferences']:
                            ref_data = {}
                            # Extract content
                            if 'content' in ref and 'text' in ref['content']:
                                ref_data['content'] = ref['content']['text']
                            # Extract location (S3 URI)
                            if 'location' in ref:
                                if 's3Location' in ref['location']:
                                    ref_data['source'] = ref['location']['s3Location'].get('uri', '')
                                elif 'type' in ref['location']:
                                    ref_data['location_type'] = ref['location']['type']
                            # Extract metadata
                            if 'metadata' in ref:
                                ref_data['metadata'] = ref['metadata']

                            retrieved_refs.append(ref_data)

                if retrieved_refs:
                    response_data['retrieved_references'] = retrieved_refs

            # Include session ID if available
            if 'sessionId' in result:
                response_data['session_id'] = result['sessionId']

            return get_json_dumps(response_data) if response_data else ""
        else:
            if arguments.get("exception") is not None:
                return get_exception_message(arguments)
            elif hasattr(arguments.get("result", {}), "error"):
                return arguments["result"].error
        
        return get_json_dumps(messages[0]) if messages else ""
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_retrieval_response: %s", str(e))
        return ""

# Helper functions for invoke_data_automation_async API
def extract_data_automation_input(kwargs):
    """Extract input information from invoke_data_automation_async parameters"""
    try:
        messages = []
        input_config = kwargs.get('inputConfiguration', {})
        output_config = kwargs.get('outputConfiguration', {})
        data_config = kwargs.get('dataAutomationConfiguration', {})
        
        input_info = {
            's3_input': input_config.get('s3Uri', ''),
            's3_output': output_config.get('s3Uri', ''),
            'project_arn': data_config.get('dataAutomationProjectArn', ''),
            'stage': data_config.get('stage', '')
        }
        messages.append({'data_automation_input': input_info})
        
        return [get_json_dumps(message) for message in messages] if messages else []
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_data_automation_input: %s", str(e))
        return []

def extract_invoke_model_messages(kwargs):
    """Extract messages from invoke_model input"""
    try:
        messages = []
        body_str = kwargs['body']
        if isinstance(body_str, (str, bytes)):
            try:
                if isinstance(body_str, bytes):
                    body_str = body_str.decode('utf-8')
                body_data = json.loads(body_str)

                # Handle different model input formats
                if 'prompt' in body_data:
                    messages.append({'user': body_data['prompt']})
                elif 'messages' in body_data:
                    for msg in body_data['messages']:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        messages.append({role: content})
                elif 'inputText' in body_data:
                    messages.append({'user': body_data['inputText']})
            except json.JSONDecodeError:
                pass

        return [get_json_dumps(message) for message in messages] if messages else []
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_invoke_model_messages: %s", str(e))
        return []


def extract_invoke_model_response(result):
    """Extract response from invoke_model result"""
    output= {}
    body = result['body']
    if hasattr(body, 'read'):
        body_bytes = body.read()
        result['body'] = BytesIO(body_bytes)
        body_str = body_bytes.decode('utf-8')
        body_data = json.loads(body_str)
        if 'choices' in body_data and len(body_data['choices']) > 0:
            choice = body_data['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                output = {'assistant': choice['message']['content']}

        return output

def update_span_from_invoke_model_response(response, instance):
    """Extract metadata from invoke_model response"""
    body = response['body']
    if hasattr(body, 'read'):
        body_bytes = body.read()
        response['body'] = BytesIO(body_bytes)
        body_str = body_bytes.decode('utf-8')
        body_data = json.loads(body_str)
        # Extract token usage if available
        if 'usage' in body_data:
            usage = body_data['usage']
            return usage