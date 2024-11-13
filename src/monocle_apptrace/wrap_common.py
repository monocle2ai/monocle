# pylint: disable=protected-access
from functools import reduce
import logging
import os
import inspect
from importlib.metadata import version
from urllib.parse import urlparse
from opentelemetry.trace import Span, Tracer
from monocle_apptrace.utils import resolve_from_alias, with_tracer_wrapper, set_embedding_model, get_embedding_model, get_attribute, get_workflow_name, get_fully_qualified_class_name, flatten_dict, get_nested_value, set_attribute, set_app_hosting_identifier_attribute

logger = logging.getLogger(__name__)
WORKFLOW_TYPE_KEY = "workflow_type"
DATA_INPUT_KEY = "data.input"
DATA_OUTPUT_KEY = "data.output"
PROMPT_INPUT_KEY = "data.input"
PROMPT_OUTPUT_KEY = "data.output"
QUERY = "question"
RESPONSE = "response"
SESSION_PROPERTIES_KEY = "session"
TYPE = "type"
PROVIDER = "provider_name"
META_DATA = 'metadata'

WORKFLOW_TYPE_MAP = {
    "llama_index": "workflow.llamaindex",
    "langchain": "workflow.langchain",
    "haystack": "workflow.haystack"
}

llm_type_map = {
    "sagemakerendpoint": "aws_sagemaker",
    "azureopenai": "azure_openai",
    "openai": "openai",
    "chatopenai": "openai",
    "azurechatopenai": "azure_openai",
    "bedrock": "aws_bedrock",
    "sagemakerllm": "aws_sagemaker",
    "chatbedrock": "aws_bedrock",
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
    

def get_nested_attribute_using(path: str, my_object: object, default=None):
    try:
        return reduce(getattr, path.split("."), my_object)
    except AttributeError:
        return default

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


    with tracer.start_as_current_span(name) as span:
        process_span(to_wrap, span, instance)
        pre_task_processing(to_wrap, instance, args, span)
        return_value = wrapped(*args, **kwargs)
        post_task_processing(to_wrap, span, return_value)

    return return_value

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

    # Check if the output_processor is a valid JSON (in Python, that means it's a dictionary)
def process_span(to_wrap, span: Span, instance):
    instance_args = {}
    set_provider_name(instance, instance_args)
    span_index = 1
    
    if is_root_span(span):
        span_index += set_workflow_attributes(to_wrap, span, span_index)
        span_index += set_app_hosting_identifier_attribute(span, span_index)
    if 'output_processor' in to_wrap:
        output_processor=to_wrap['output_processor']
        if isinstance(output_processor, dict) and len(output_processor) > 0:
            if 'type' in output_processor:
                span.set_attribute("span.type", output_processor['type'])
            else:
                logger.warning("type of span not found or incorrect written in entity json")
            count = 0
            if 'attributes' in output_processor:
                count = len(output_processor["attributes"])
                span.set_attribute("entity.count", count)
                for processors in output_processor["attributes"]:
                    for processor in processors:
                        attribute = processor.get('attribute')
                        accessor = processor.get('accessor')

                        if attribute and accessor:
                            attribute_name = f"entity.{span_index}.{attribute}"
                            try:
                                result = eval(accessor)(instance, instance_args)
                                if result and isinstance(result, str):
                                    span.set_attribute(attribute_name, result)
                            except Exception as e:
                                logger.error(f"Error processing accessor: {e}")
                        else:
                            logger.warning(f"{' and '.join([key for key in ['attribute', 'accessor'] if not processor.get(key)])} not found or incorrect in entity JSON")
                    span_index += 1
            else:
                logger.warning("attributes not found or incorrect written in entity json")
                span.set_attribute("span.count", count)

        else:
            logger.warning("empty or entities json is not in correct format")
    else:
        if span_index > 1:
            span.set_attribute("entity.count", span_index-1)

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
                sdk_version = version("monocle-apptrace")
                span.set_attribute("monocle-sdk.version",sdk_version)
            except:
                logger.warning(f"Exception finding monocle_apptrace version.")
            update_span_with_prompt_input(to_wrap=to_wrap, wrapped_args=args, span=span)

        update_span_with_context_input(to_wrap=to_wrap, wrapped_args=args, span=span)
    except:
        logger.exception("exception in pre_task_processing")


@with_tracer_wrapper
async def atask_wrapper(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
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
        process_span(to_wrap, span, instance)
        pre_task_processing(to_wrap, instance, args, span)
        return_value = await wrapped(*args, **kwargs)
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
        name =  get_fully_qualified_class_name(instance)
    with tracer.start_as_current_span(name) as span:
        process_span(to_wrap, span, instance)
        pre_task_processing(to_wrap, instance, args, span)
        return_value = await wrapped(*args, **kwargs)
        post_task_processing(to_wrap, span, return_value)
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
        name =  get_fully_qualified_class_name(instance)

    with tracer.start_as_current_span(name) as span:
        process_span(to_wrap, span, instance)
        pre_task_processing(to_wrap, instance, args, span)
        return_value = wrapped(*args, **kwargs)
        post_task_processing(to_wrap, span, return_value)
        update_span_from_llm_response(response=return_value, span=span, instance=instance)

    return return_value

def set_provider_name(instance, instance_args: dict):
    provider_url = ""

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
    parsed_provider_url = ""
    try:
        if len(provider_url) > 0:
            parsed_provider_url = urlparse(provider_url).hostname
    except:
        pass
    if parsed_provider_url or provider_url:
        instance_args[PROVIDER] = parsed_provider_url or provider_url
    try:
        llm_type = llm_type_map.get(type(instance).__name__.lower())
        instance_args["llm_type"] = llm_type
    except:
        pass


def is_root_span(curr_span: Span) -> bool:
    if (curr_span is not None and hasattr(curr_span, "parent")):
        return curr_span.parent is None
    return False


def get_input_from_args(chain_args):
    if len(chain_args) > 0 and isinstance(chain_args[0], str):
        return chain_args[0]
    return ""


def update_span_from_llm_response(response, span: Span, instance):
    if (response is not None and isinstance(response, dict) and "meta" in response) or (response is not None and hasattr(response, "response_metadata")):
        token_usage = None
        if (response is not None and isinstance(response, dict) and "meta" in response):  # haystack
            token_usage = response["meta"][0]["usage"]

        if (response is not None and hasattr(response, "response_metadata")):
            response_metadata = response.response_metadata
            token_usage = response_metadata.get("token_usage")

        meta_dict = {}
        if token_usage is not None:
            temperature = instance.__dict__.get("temperature", None)
            meta_dict.update({"temperature": temperature})
            meta_dict.update({"completion_tokens": token_usage.get("completion_tokens")})
            meta_dict.update({"prompt_tokens": token_usage.get("prompt_tokens")})
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
    try:
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
    except Exception as e:
        logger.warning(f"Error updating span with context input {e}")


def update_span_with_context_output(to_wrap, return_value, span: Span):
    try:
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
    except Exception as e:
        logger.warning(f"Error updating span with context output {e}")


def update_span_with_prompt_input(to_wrap, wrapped_args, span: Span):
    try:
        input_arg_text = wrapped_args[0]

        prompt_inputs = get_nested_value(input_arg_text, ['prompt_builder'])
        if prompt_inputs is not None:  # haystack
            input_arg_text = flatten_dict(prompt_inputs)
            span.add_event(PROMPT_INPUT_KEY, input_arg_text)
        elif isinstance(input_arg_text, dict):
            span.add_event(PROMPT_INPUT_KEY, input_arg_text)
        else:
            span.add_event(PROMPT_INPUT_KEY, {QUERY: input_arg_text})
    except Exception as e:
        logger.warning(f"Error updating span with prompt input {e}")


def update_span_with_prompt_output(to_wrap, wrapped_args, span: Span):
    try:
        package_name: str = to_wrap.get('package')

        if "llama_index.core.base.base_query_engine" in package_name:
            span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: wrapped_args.response})
        elif "haystack.core.pipeline.pipeline" in package_name:
            resp = get_nested_value(wrapped_args, ['llm', 'replies'])
            if resp is not None:
                span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: resp})
        elif isinstance(wrapped_args, str):
            span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: wrapped_args})
        elif isinstance(wrapped_args, dict):
            span.add_event(PROMPT_OUTPUT_KEY,  wrapped_args)
    except Exception as e:
        logger.warning(f"Error updating span with prompt output {e}")
