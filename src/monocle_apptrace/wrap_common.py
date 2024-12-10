# pylint: disable=protected-access
import logging
import os
import inspect
from importlib.metadata import version
from urllib.parse import urlparse
from opentelemetry.trace import Tracer
from opentelemetry.sdk.trace import Span
from monocle_apptrace.utils import resolve_from_alias, with_tracer_wrapper, get_embedding_model, get_attribute, get_workflow_name, set_embedding_model, set_app_hosting_identifier_attribute
from monocle_apptrace.utils import set_attribute, get_vectorstore_deployment
from monocle_apptrace.utils import get_fully_qualified_class_name, get_nested_value
from monocle_apptrace.message_processing import extract_messages, extract_assistant_message
from functools import wraps

logger = logging.getLogger(__name__)
WORKFLOW_TYPE_KEY = "workflow_type"
DATA_INPUT_KEY = "data.input"
DATA_OUTPUT_KEY = "data.output"
PROMPT_INPUT_KEY = "data.input"
PROMPT_OUTPUT_KEY = "data.output"
QUERY = "input"
RESPONSE = "response"
SESSION_PROPERTIES_KEY = "session"
INFRA_SERVICE_KEY = "infra_service_name"

TYPE = "type"
PROVIDER = "provider_name"
EMBEDDING_MODEL = "embedding_model"
VECTOR_STORE = 'vector_store'
META_DATA = 'metadata'

WORKFLOW_TYPE_MAP = {
    "llama_index": "workflow.llamaindex",
    "langchain": "workflow.langchain",
    "haystack": "workflow.haystack"
}


def get_embedding_model_haystack(instance):
    try:
        if hasattr(instance, 'get_component'):
            text_embedder = instance.get_component('text_embedder')
            if text_embedder and hasattr(text_embedder, 'model'):
                # Set the embedding model attribute
                return text_embedder.model
    except:
        pass

    return None

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
        name = get_fully_qualified_class_name(instance)

    if 'haystack.core.pipeline.pipeline' in to_wrap['package']:
        embedding_model = get_embedding_model_haystack(instance)
        set_embedding_model(embedding_model)
        inputs = set()
        workflow_input = get_workflow_input(args, inputs)
        set_attribute(DATA_INPUT_KEY, workflow_input)

    if to_wrap.get('skip_span'):
        return_value = wrapped(*args, **kwargs)
        botocore_processor(tracer, to_wrap, wrapped, instance, args, kwargs, return_value)
        return return_value

    with tracer.start_as_current_span(name) as span:
        pre_task_processing(to_wrap, instance, args, span)
        return_value = wrapped(*args, **kwargs)
        process_span(to_wrap, span, instance, args, kwargs, return_value)
        post_task_processing(to_wrap, span, return_value)

    return return_value

def botocore_processor(tracer, to_wrap, wrapped, instance, args, kwargs,return_value):
    if kwargs.get("service_name") == "sagemaker-runtime":
        return_value.invoke_endpoint = _instrumented_endpoint_invoke(to_wrap,return_value,return_value.invoke_endpoint,tracer)

def _instrumented_endpoint_invoke(to_wrap, instance, fn, tracer):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):

        with tracer.start_as_current_span("botocore-sagemaker-invoke-endpoint") as span:
            response = fn(*args, **kwargs)
            process_span(to_wrap, span, instance=instance,args=args, kwargs=kwargs, return_value=response)
            return response

    return with_instrumentation

def get_workflow_input(args, inputs):
    if args is not None and len(args) > 0:
        for value in args[0].values():
            for text in value.values():
                inputs.add(text)

    workflow_input: str = ""

    if inputs is not None and len(inputs) > 0:
        for input_str in inputs:
            workflow_input = workflow_input + input_str
    return workflow_input

def process_span(to_wrap, span, instance, args, kwargs, return_value):
    # Check if the output_processor is a valid JSON (in Python, that means it's a dictionary)
    instance_args = {}
    set_provider_name(instance, instance_args)
    span_index = 0
    if is_root_span(span):
        span_index += set_workflow_attributes(to_wrap, span, span_index+1)
        span_index += set_app_hosting_identifier_attribute(span, span_index+1)
    if 'output_processor' in to_wrap:
        output_processor=to_wrap['output_processor']
        if isinstance(output_processor, dict) and len(output_processor) > 0:
            if 'type' in output_processor:
                span.set_attribute("span.type", output_processor['type'])
            else:
                logger.warning("type of span not found or incorrect written in entity json")
            if 'attributes' in output_processor:
                for processors in output_processor["attributes"]:
                    for processor in processors:
                        attribute = processor.get('attribute')
                        accessor = processor.get('accessor')

                        if attribute and accessor:
                            attribute_name = f"entity.{span_index+1}.{attribute}"
                            try:
                                arguments = {"instance":instance, "args":args, "kwargs":kwargs, "output":return_value}
                                result = eval(accessor)(arguments)
                                if result and isinstance(result, str):
                                    span.set_attribute(attribute_name, result)
                            except Exception as e:
                                logger.error(f"Error processing accessor: {e}")
                        else:
                            logger.warning(f"{' and '.join([key for key in ['attribute', 'accessor'] if not processor.get(key)])} not found or incorrect in entity JSON")
                    span_index += 1
            else:
                logger.warning("attributes not found or incorrect written in entity json")
            if 'events' in output_processor:
                events = output_processor['events']
                arguments = {"instance": instance, "args": args, "kwargs": kwargs, "output": return_value}
                accessor_mapping = {
                    "arguments": arguments,
                    "response": return_value
                }
                for event in events:
                    event_name = event.get("name")
                    event_attributes = {}
                    attributes = event.get("attributes", [])
                    for attribute in attributes:
                        attribute_key = attribute.get("attribute")
                        accessor = attribute.get("accessor")
                        if accessor:
                            try:
                                accessor_function = eval(accessor)
                                for keyword, value in accessor_mapping.items():
                                    if keyword in accessor:
                                        evaluated_val = accessor_function(value)
                                        if isinstance(evaluated_val, list):
                                            evaluated_val = [str(d) for d in evaluated_val]
                                        event_attributes[attribute_key] = evaluated_val
                            except Exception as e:
                                logger.error(f"Error evaluating accessor for attribute '{attribute_key}': {e}")
                    span.add_event(name=event_name, attributes=event_attributes)

        else:
            logger.warning("empty or entities json is not in correct format")
    if span_index > 0:
        span.set_attribute("entity.count", span_index)

def set_workflow_attributes(to_wrap, span: Span, span_index):
    return_value = 1
    workflow_name = get_workflow_name(span=span)
    if workflow_name:
        span.set_attribute("span.type", "workflow")
        span.set_attribute(f"entity.{span_index}.name", workflow_name)
        # workflow type
    package_name = to_wrap.get('package')
    workflow_type_set = False
    for (package, workflow_type) in WORKFLOW_TYPE_MAP.items():
        if (package_name is not None and package in package_name):
            span.set_attribute(f"entity.{span_index}.type", workflow_type)
            workflow_type_set = True
    if not workflow_type_set:
        span.set_attribute(f"entity.{span_index}.type", "workflow.generic")
    return return_value

def post_task_processing(to_wrap, span, return_value):
    try:
        update_span_with_context_output(to_wrap=to_wrap, return_value=return_value, span=span)

        if is_root_span(span):
            update_span_with_prompt_output(to_wrap=to_wrap, wrapped_args=return_value, span=span)
    except:
        logger.exception("exception in post_task_processing")


def pre_task_processing(to_wrap, instance, args, span):
    try:
        if is_root_span(span):
            try:
                sdk_version = version("monocle_apptrace")
                span.set_attribute("monocle_apptrace.version", sdk_version)
            except:
                logger.warning(f"Exception finding monocle-apptrace version.")
            update_span_with_prompt_input(to_wrap=to_wrap, wrapped_args=args, span=span)
        update_span_with_context_input(to_wrap=to_wrap, wrapped_args=args, span=span)
    except:
        logger.exception("exception in pre_task_processing")


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
        name = get_fully_qualified_class_name(instance)
    if 'haystack.core.pipeline.pipeline' in to_wrap['package']:
        embedding_model = get_embedding_model_haystack(instance)
        set_embedding_model(embedding_model)
        inputs = set()
        workflow_input = get_workflow_input(args, inputs)
        set_attribute(DATA_INPUT_KEY, workflow_input)

    with tracer.start_as_current_span(name) as span:
        pre_task_processing(to_wrap, instance, args, span)
        return_value = await wrapped(*args, **kwargs)
        process_span(to_wrap, span, instance, args, kwargs, return_value)
        post_task_processing(to_wrap, span, return_value)

    return return_value


@with_tracer_wrapper
async def allm_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
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
        name = get_fully_qualified_class_name(instance)
    with tracer.start_as_current_span(name) as span:
        provider_name, inference_endpoint = get_provider_name(instance)
        return_value = await wrapped(*args, **kwargs)
        kwargs.update({"provider_name": provider_name, "inference_endpoint": inference_endpoint or getattr(instance, 'endpoint', None)})
        process_span(to_wrap, span, instance, args, kwargs, return_value)
        update_span_from_llm_response(response=return_value, span=span, instance=instance)

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
        name = get_fully_qualified_class_name(instance)

    with tracer.start_as_current_span(name) as span:
        provider_name, inference_endpoint = get_provider_name(instance)
        return_value = wrapped(*args, **kwargs)
        kwargs.update({"provider_name": provider_name, "inference_endpoint": inference_endpoint or getattr(instance, 'endpoint', None)})
        process_span(to_wrap, span, instance, args, kwargs, return_value)
        update_span_from_llm_response(response=return_value, span=span, instance=instance)

    return return_value


def update_llm_endpoint(curr_span: Span, instance):
    # Lambda to set attributes if values are not None
    __set_span_attribute_if_not_none = lambda span, **kwargs: [
        span.set_attribute(k, v) for k, v in kwargs.items() if v is not None
    ]

    triton_llm_endpoint = os.environ.get("TRITON_LLM_ENDPOINT")
    if triton_llm_endpoint is not None and len(triton_llm_endpoint) > 0:
        curr_span.set_attribute("server_url", triton_llm_endpoint)
    else:
        # Get temperature if present
        temp_val = instance.__dict__.get("temperature")

        # Resolve values for model name, deployment, and inference endpoint
        model_name = resolve_from_alias(instance.__dict__, ["model", "model_name"])
        deployment_name = resolve_from_alias(instance.__dict__,
                                             ["engine", "azure_deployment", "deployment_name", "deployment_id",
                                              "deployment"])
        inference_ep = resolve_from_alias(instance.__dict__, ["azure_endpoint", "api_base"])

        # Use the lambda to set attributes conditionally
        __set_span_attribute_if_not_none(
            curr_span,
            temperature=temp_val,
            model_name=model_name,
            az_openai_deployment=deployment_name,
            inference_endpoint=inference_ep
        )


def get_provider_name(instance):
    provider_url = ""
    inference_endpoint = ""
    parsed_provider_url = ""
    try:
        base_url = getattr(instance.client._client, "base_url", None)
        if base_url:
            if isinstance(getattr(base_url, "host", None), str):
                provider_url = base_url.host
            inference_endpoint = base_url if isinstance(base_url, str) else str(base_url)
    except:
        pass

    try:
        if isinstance(instance.client.meta.endpoint_url, str):
            inference_endpoint = instance.client.meta.endpoint_url
    except:
        pass

    api_base = getattr(instance, "api_base", None)
    if isinstance(api_base, str):
        provider_url = api_base

    # Handle inference endpoint for Mistral AI (llamaindex)
    sdk_config = getattr(instance, "_client", None)
    if sdk_config and hasattr(sdk_config, "sdk_configuration"):
        inference_endpoint = getattr(sdk_config.sdk_configuration, "server_url", inference_endpoint)

    if provider_url:
        try:
            parsed_provider_url = urlparse(provider_url)
        except:
            pass

    return parsed_provider_url.hostname if parsed_provider_url else provider_url, inference_endpoint


def set_provider_name(instance, instance_args: dict):
    provider_url = ""
    parsed_provider_url = ""
    try:
        if isinstance(instance.client._client.base_url.host, str):
            provider_url = instance.client._client.base_url.host
    except:
        pass

    try:
        if isinstance(instance.api_base, str):
            provider_url = instance.api_base
    except:
        pass
    try:
        if len(provider_url) > 0:
            parsed_provider_url = urlparse(provider_url).hostname
    except:
        pass
    if parsed_provider_url or provider_url:
        instance_args[PROVIDER] = parsed_provider_url or provider_url


def is_root_span(curr_span: Span) -> bool:
    return curr_span.parent is None


def get_input_from_args(chain_args):
    if len(chain_args) > 0 and isinstance(chain_args[0], str):
        return chain_args[0]
    return ""


def update_span_from_llm_response(response, span: Span, instance):
    if (response is not None and isinstance(response, dict) and "meta" in response) or (
            response is not None and hasattr(response, "response_metadata")):
        token_usage = None
        if (response is not None and isinstance(response, dict) and "meta" in response):  # haystack
            token_usage = response["meta"][0]["usage"]

        if (response is not None and hasattr(response, "response_metadata")):
            if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
                token_usage = response.usage_metadata
            else:
                response_metadata = response.response_metadata
                token_usage = response_metadata.get("token_usage")

        meta_dict = {}
        if token_usage is not None:
            temperature = instance.__dict__.get("temperature", None)
            meta_dict.update({"temperature": temperature})
            meta_dict.update({"completion_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens")})
            meta_dict.update({"prompt_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens")})
            meta_dict.update({"total_tokens": token_usage.get("total_tokens")})
            span.add_event(META_DATA, meta_dict)
    # extract token usage from llamaindex openai
    if (response is not None and hasattr(response, "raw")):
        try:
            meta_dict = {}
            if response.raw is not None:
                token_usage = response.raw.get("usage") if isinstance(response.raw, dict) else getattr(response.raw,
                                                                                                       "usage", None)
                if token_usage is not None:
                    temperature = instance.__dict__.get("temperature", None)
                    meta_dict.update({"temperature": temperature})
                    if getattr(token_usage, "completion_tokens", None):
                        meta_dict.update({"completion_tokens": getattr(token_usage, "completion_tokens")})
                    if getattr(token_usage, "prompt_tokens", None):
                        meta_dict.update({"prompt_tokens": getattr(token_usage, "prompt_tokens")})
                    if getattr(token_usage, "total_tokens", None):
                        meta_dict.update({"total_tokens": getattr(token_usage, "total_tokens")})
                    span.add_event(META_DATA, meta_dict)
        except AttributeError:
            token_usage = None


def update_workflow_type(to_wrap, span: Span):
    package_name = to_wrap.get('package')

    for (package, workflow_type) in WORKFLOW_TYPE_MAP.items():
        if (package_name is not None and package in package_name):
            span.set_attribute(WORKFLOW_TYPE_KEY, workflow_type)


def update_span_with_context_input(to_wrap, wrapped_args, span: Span):
    package_name: str = to_wrap.get('package')
    input_arg_text = ""
    if "langchain_core.retrievers" in package_name and len(wrapped_args) > 0:
        input_arg_text += wrapped_args[0]
    if "llama_index.core.indices.base_retriever" in package_name and len(wrapped_args) > 0:
        input_arg_text += wrapped_args[0].query_str
    if "haystack.components.retrievers.in_memory" in package_name:
        input_arg_text += get_attribute(DATA_INPUT_KEY)
    if input_arg_text:
        span.add_event(DATA_INPUT_KEY, {QUERY: input_arg_text})


def update_span_with_context_output(to_wrap, return_value, span: Span):
    package_name: str = to_wrap.get('package')
    output_arg_text = ""
    if "langchain_core.retrievers" in package_name:
        output_arg_text += " ".join([doc.page_content for doc in return_value if hasattr(doc, 'page_content')])
        if len(output_arg_text) > 100:
            output_arg_text = output_arg_text[:100] + "..."
    if "llama_index.core.indices.base_retriever" in package_name and len(return_value) > 0:
        output_arg_text += return_value[0].text
    if "haystack.components.retrievers.in_memory" in package_name:
        output_arg_text += " ".join([doc.content for doc in return_value['documents']])
        if len(output_arg_text) > 100:
            output_arg_text = output_arg_text[:100] + "..."
    if output_arg_text:
        span.add_event(DATA_OUTPUT_KEY, {RESPONSE: output_arg_text})


def update_span_with_prompt_input(to_wrap, wrapped_args, span: Span):
    input_arg_text = wrapped_args[0]

    prompt_inputs = get_nested_value(input_arg_text, ['prompt_builder', 'question'])
    if prompt_inputs is not None:  # haystack
        span.add_event(PROMPT_INPUT_KEY, {QUERY: prompt_inputs})
    elif isinstance(input_arg_text, dict):
        span.add_event(PROMPT_INPUT_KEY, {QUERY: input_arg_text['input']})
    else:
        span.add_event(PROMPT_INPUT_KEY, {QUERY: input_arg_text})


def update_span_with_prompt_output(to_wrap, wrapped_args, span: Span):
    package_name: str = to_wrap.get('package')

    if "llama_index.core.base.base_query_engine" in package_name:
        span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: wrapped_args.response})
    elif "haystack.core.pipeline.pipeline" in package_name:
        resp = get_nested_value(wrapped_args, ['llm', 'replies'])
        if resp is not None:
            if isinstance(resp, list) and hasattr(resp[0], 'content'):
                span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: resp[0].content})
            else:
                span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: resp[0]})
    elif isinstance(wrapped_args, str):
        span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: wrapped_args})
    elif isinstance(wrapped_args, dict):
        span.add_event(PROMPT_OUTPUT_KEY, wrapped_args)
