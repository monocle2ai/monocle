"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
import json
from io import BytesIO
from functools import wraps
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler

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
        return [str(d) for d in messages]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(response):
    try:
        if "Body" in response and hasattr(response['Body'], "_raw_stream"):
            raw_stream = getattr(response['Body'], "_raw_stream")
            if hasattr(raw_stream, "data"):
                response_bytes = getattr(raw_stream, "data")
                response_str = response_bytes.decode('utf-8')
                response_dict = json.loads(response_str)
                response['Body'] = BytesIO(response_bytes)
                return [response_dict["answer"]]
        if "output" in response:
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])
            if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                reply = content[0]["text"]
                return [reply]
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


def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None


def botocore_processor(tracer, to_wrap, wrapped, return_value, args, kwargs):
    service_name = kwargs.get("service_name")
    service_method_mapping = {
        "sagemaker-runtime": "invoke_endpoint",
        "bedrock-runtime": "converse",
    }
    if service_name in service_method_mapping:
        method_name = service_method_mapping[service_name]
        original_method = getattr(return_value, method_name, None)

        if original_method:
            instrumented_method = _instrumented_endpoint_invoke(
                to_wrap, wrapped,return_value, original_method, tracer, service_name
            )
            setattr(return_value, method_name, instrumented_method)

def _instrumented_endpoint_invoke(to_wrap,wrapped, instance, fn, tracer,service_name):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        span_name="botocore-"+service_name+"-invoke-endpoint"
        handler = SpanHandler()
        with tracer.start_as_current_span(span_name) as span:
            response = fn(*args, **kwargs)
            handler.hydrate_span(to_wrap, span=span,wrapped=wrapped, instance=instance,args=args, kwargs=kwargs, result=response)
            return response

    return with_instrumentation


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
