# Copyright (C) Okahu Inc 2023-2024. All rights reserved

import logging
import os

from opentelemetry.trace import Span, Tracer

from monocle_apptrace.utils import resolve_from_alias, with_tracer_wrapper

logger = logging.getLogger(__name__)
WORKFLOW_TYPE_KEY = "workflow_type"
CONTEXT_INPUT_KEY = "context_input"
CONTEXT_OUTPUT_KEY = "context_output"
PROMPT_INPUT_KEY = "input"
PROMPT_OUTPUT_KEY = "output"
QUERY = "question"
RESPONSE = "response"
TAGS = "tags"
CONTEXT_PROPERTIES_KEY = "workflow_context_properties"


WORKFLOW_TYPE_MAP = {
    "llama_index": "workflow.llamaindex",
    "langchain": "workflow.langchain",
    "haystack": "workflow.haystack"
}

@with_tracer_wrapper
def task_wrapper(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""

    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    if hasattr(instance, "name") and instance.name:
        name = f"{to_wrap.get('span_name')}.{instance.name.lower()}"
    elif to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = f"langchain.task.{instance.__class__.__name__}"
    kind = to_wrap.get("kind")

    with tracer.start_as_current_span(name) as span:
        if is_root_span(span):
            update_span_with_prompt_input(to_wrap=to_wrap, wrapped_args=args, span=span)

        #capture the tags attribute of the instance if present, else ignore 
        try:
            span.set_attribute(TAGS, getattr(instance, TAGS))
        except AttributeError:
            pass    
        update_span_with_context_input(to_wrap=to_wrap, wrapped_args=args, span=span)
        return_value = wrapped(*args, **kwargs)
        update_span_with_context_output(to_wrap=to_wrap, return_value=return_value, span=span)

        if is_root_span(span):
            workflow_name = span.resource.attributes.get("service.name")
            span.set_attribute("workflow_name",workflow_name)
            update_span_with_prompt_output(to_wrap=to_wrap, wrapped_args=return_value, span=span)
            update_workflow_type(to_wrap, span)

    return return_value

@with_tracer_wrapper
async def atask_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""

    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    if hasattr(instance, "name") and instance.name:
        name = f"{to_wrap.get('span_name')}.{instance.name.lower()}"
    elif to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = f"langchain.task.{instance.__class__.__name__}"
    kind = to_wrap.get("kind")
    with tracer.start_as_current_span(name) as span:
        return_value = await wrapped(*args, **kwargs)

    return return_value

@with_tracer_wrapper
async def allm_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    if to_wrap.get("span_name_getter"):
        name = to_wrap.get("span_name_getter")(instance)

    elif hasattr(instance, "name") and instance.name:
        name = f"{to_wrap.get('span_name')}.{instance.name.lower()}"
    elif to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = f"langchain.task.{instance.__class__.__name__}"
    kind = to_wrap.get("kind")
    with tracer.start_as_current_span(name) as span:
        update_llm_endpoint(curr_span= span, instance=instance)

        return_value = await wrapped(*args, **kwargs)

    return return_value

@with_tracer_wrapper
def llm_wrapper(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):

    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    if callable(to_wrap.get("span_name_getter")):
        name = to_wrap.get("span_name_getter")(instance)

    elif hasattr(instance, "name") and instance.name:
        name = f"{to_wrap.get('span_name')}.{instance.name.lower()}"
    elif to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = f"langchain.task.{instance.__class__.__name__}"
    kind = to_wrap.get("kind")
    with tracer.start_as_current_span(name) as span:
        update_llm_endpoint(curr_span= span, instance=instance)

        return_value = wrapped(*args, **kwargs)
        update_span_from_llm_response(response = return_value, span = span)

    return return_value

def update_llm_endpoint(curr_span: Span, instance):
    triton_llm_endpoint = os.environ.get("TRITON_LLM_ENDPOINT")
    if triton_llm_endpoint is not None and len(triton_llm_endpoint) > 0:
        curr_span.set_attribute("server_url", triton_llm_endpoint)
    else:
        if 'temperature' in instance.__dict__:
            temp_val = instance.__dict__.get("temperature")
            curr_span.set_attribute("temperature", temp_val)
        # handling for model name
        model_name =  resolve_from_alias(instance.__dict__ , ["model","model_name"])
        curr_span.set_attribute("openai_model_name", model_name)
        # handling AzureOpenAI deployment
        deployment_name =  resolve_from_alias(instance.__dict__ , [ "engine", "azure_deployment", 
                                                                   "deployment_name", "deployment_id", "deployment"])
        curr_span.set_attribute("az_openai_deployment", deployment_name)
        # handling the inference endpoint
        inference_ep = resolve_from_alias(instance.__dict__,["azure_endpoint","api_base"])
        curr_span.set_attribute("inference_endpoint",inference_ep)

def is_root_span(curr_span: Span) -> bool:
    return curr_span.parent == None

def get_input_from_args(chain_args):
    if len(chain_args) > 0 and type(chain_args[0]) == str:
        return chain_args[0]
    return ""

def update_span_from_llm_response(response, span: Span):
    # extract token uasge from langchain openai
    if (response is not None and hasattr(response, "response_metadata")):
        response_metadata = response.response_metadata
        token_usage = response_metadata.get("token_usage")
        if token_usage is not None:
            span.set_attribute("completion_tokens", token_usage.get("completion_tokens"))
            span.set_attribute("prompt_tokens", token_usage.get("prompt_tokens"))
            span.set_attribute("total_tokens", token_usage.get("total_tokens"))
    # extract token usage from llamaindex openai
    if(response is not None and hasattr(response, "raw")):
        if response.raw is not None:
            token_usage = response.raw.get("usage")
            if token_usage is not None:
                if(hasattr(token_usage, "completion_tokens")):
                    span.set_attribute("completion_tokens", token_usage.completion_tokens)
                if(hasattr(token_usage, "prompt_tokens")):
                    span.set_attribute("prompt_tokens", token_usage.prompt_tokens)
                if(hasattr(token_usage, "total_tokens")):
                    span.set_attribute("total_tokens", token_usage.total_tokens)

def update_workflow_type(to_wrap, span: Span):
    package_name = to_wrap.get('package')

    for (package, workflow_type) in WORKFLOW_TYPE_MAP.items():
        if(package_name is not None and package in package_name):
            span.set_attribute(WORKFLOW_TYPE_KEY, workflow_type)

def update_span_with_context_input(to_wrap, wrapped_args ,span: Span):
    package_name: str = to_wrap.get('package')
    if("langchain_core.retrievers" in package_name):
        input_arg_text = wrapped_args[0]
        span.add_event(CONTEXT_INPUT_KEY, {QUERY:input_arg_text})
    if("llama_index.core.indices.base_retriever" in package_name):
        input_arg_text = wrapped_args[0].query_str
        span.add_event(CONTEXT_INPUT_KEY, {QUERY:input_arg_text})

def update_span_with_context_output(to_wrap, return_value ,span: Span):
    package_name: str = to_wrap.get('package')
    if("llama_index.core.indices.base_retriever" in package_name):
        output_arg_text = return_value[0].text
        span.add_event(CONTEXT_OUTPUT_KEY, {RESPONSE:output_arg_text})

def update_span_with_prompt_input(to_wrap, wrapped_args ,span: Span):
    input_arg_text = wrapped_args[0]
    
    if isinstance(input_arg_text, dict):
        span.add_event(PROMPT_INPUT_KEY,input_arg_text)
    else:    
        span.add_event(PROMPT_INPUT_KEY,{QUERY:input_arg_text})

def update_span_with_prompt_output(to_wrap, wrapped_args ,span: Span):
    package_name: str = to_wrap.get('package')
    if(isinstance(wrapped_args, str)):
        span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE:wrapped_args})
    if("llama_index.core.base.base_query_engine" in package_name):
        span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE:wrapped_args.response})
