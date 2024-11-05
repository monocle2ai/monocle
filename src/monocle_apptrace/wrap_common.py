# pylint: disable=protected-access
import logging
import os
import inspect
from urllib.parse import urlparse
from opentelemetry.trace import Span, Tracer
from monocle_apptrace.utils import resolve_from_alias, update_span_with_infra_name, with_tracer_wrapper, get_embedding_model, get_attribute, get_workflow_name
from monocle_apptrace.utils import set_attribute
from opentelemetry.context import get_value, attach, set_value
logger = logging.getLogger(__name__)
WORKFLOW_TYPE_KEY = "workflow_type"
DATA_INPUT_KEY = "data.input"
DATA_OUTPUT_KEY = "data.output"
PROMPT_INPUT_KEY = "data.input"
PROMPT_OUTPUT_KEY = "data.output"
QUERY = "question"
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

def get_embedding_model_for_vectorstore(instance):
    # Handle Langchain or other frameworks where vectorstore exists
    if hasattr(instance, 'vectorstore'):
        vectorstore_dict = instance.vectorstore.__dict__

        # Use inspect to check if the embedding function is from Sagemaker
        if 'embedding_func' in vectorstore_dict:
            embedding_func = vectorstore_dict['embedding_func']
            class_name = embedding_func.__class__.__name__
            file_location = inspect.getfile(embedding_func.__class__)

            # Check if the class is SagemakerEndpointEmbeddings
            if class_name == 'SagemakerEndpointEmbeddings' and 'langchain_community' in file_location:
                # Set embedding_model as endpoint_name if it's Sagemaker
                if hasattr(embedding_func, 'endpoint_name'):
                    return embedding_func.endpoint_name

        # Default to the regular embedding model if not Sagemaker
        return instance.vectorstore.embeddings.model

    # Handle llama_index where _embed_model is present
    if hasattr(instance, '_embed_model') and hasattr(instance._embed_model, 'model_name'):
        return instance._embed_model.model_name

    # Fallback if no specific model is found
    return "Unknown Embedding Model"


framework_vector_store_mapping = {
    'langchain_core.retrievers': lambda instance: {
        'provider': type(instance.vectorstore).__name__,
        'embedding_model': get_embedding_model_for_vectorstore(instance),
        'type': VECTOR_STORE,
    },
    'llama_index.core.indices.base_retriever': lambda instance: {
        'provider': type(instance._vector_store).__name__,
        'embedding_model': get_embedding_model_for_vectorstore(instance),
        'type': VECTOR_STORE,
    },
    'haystack.components.retrievers.in_memory': lambda instance: {
        'provider': instance.__dict__.get("document_store").__class__.__name__,
        'embedding_model': get_embedding_model(),
        'type': VECTOR_STORE,
    },
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

    with tracer.start_as_current_span(name) as span:
        process_span(to_wrap, span, instance, args)
        pre_task_processing(to_wrap, instance, args, span)
        return_value = wrapped(*args, **kwargs)
        post_task_processing(to_wrap, span, return_value)

    return return_value


def process_span(to_wrap, span, instance, args):
    # Check if the output_processor is a valid JSON (in Python, that means it's a dictionary)
    span_index = 1
    if is_root_span(span):
        workflow_name = get_workflow_name(span)
        if workflow_name:
            span.set_attribute(f"entity.{span_index}.name", workflow_name)
        # workflow type
        package_name = to_wrap.get('package')
        for (package, workflow_type) in WORKFLOW_TYPE_MAP.items():
            if (package_name is not None and package in package_name):
                span.set_attribute(f"entity.{span_index}.type", workflow_type)
        span_index += 1
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
                span_index = 1
                for processors in output_processor["attributes"]:
                    for processor in processors:
                        attribute = processor.get('attribute')
                        accessor = processor.get('accessor')

                        if attribute and accessor:
                            attribute_name = f"entity.{span_index}.{attribute}"
                            try:
                                result = eval(accessor)(instance, args)
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
            update_span_with_prompt_input(to_wrap=to_wrap, wrapped_args=args, span=span)
            update_span_with_infra_name(span, INFRA_SERVICE_KEY)

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
        name = f"langchain.task.{instance.__class__.__name__}"
    with tracer.start_as_current_span(name) as span:
        process_span(to_wrap, span, instance, args)
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
        name = f"langchain.task.{instance.__class__.__name__}"
    with tracer.start_as_current_span(name) as span:
        if 'haystack.components.retrievers' in to_wrap['package'] and 'haystack.retriever' in span.name:
            input_arg_text = get_attribute(DATA_INPUT_KEY)
            span.add_event(DATA_INPUT_KEY, {QUERY: input_arg_text})
        provider_name, inference_endpoint = get_provider_name(instance)
        instance_args = {"provider_name": provider_name, "inference_endpoint": inference_endpoint}

        process_span(to_wrap, span, instance, instance_args)

        return_value = await wrapped(*args, **kwargs)
        if 'haystack.components.retrievers' in to_wrap['package'] and 'haystack.retriever' in span.name:
            update_span_with_context_output(to_wrap=to_wrap, return_value=return_value, span=span)
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
        name = f"langchain.task.{instance.__class__.__name__}"

    with tracer.start_as_current_span(name) as span:
        if 'haystack.components.retrievers' in to_wrap['package'] and 'haystack.retriever' in span.name:
            input_arg_text = get_attribute(DATA_INPUT_KEY)
            span.add_event(DATA_INPUT_KEY, {QUERY: input_arg_text})
        provider_name, inference_endpoint = get_provider_name(instance)
        instance_args = {"provider_name": provider_name, "inference_endpoint": inference_endpoint}

        process_span(to_wrap, span, instance, instance_args)

        return_value = wrapped(*args, **kwargs)
        if 'haystack.components.retrievers' in to_wrap['package'] and 'haystack.retriever' in span.name:
            update_span_with_context_output(to_wrap=to_wrap, return_value=return_value, span=span)
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
    try:
        if isinstance(instance.client._client.base_url.host, str):
            provider_url = instance.client._client.base_url.host
        if isinstance(instance.client._client.base_url, str):
            inference_endpoint = instance.client._client.base_url
        else:
            inference_endpoint = str(instance.client._client.base_url)
    except:
        pass

    try:
        if isinstance(instance.api_base, str):
            provider_url = instance.api_base
    except:
        pass

    try:
        if len(provider_url) > 0:
            parsed_provider_url = urlparse(provider_url)
    except:
        pass
    return parsed_provider_url.hostname or provider_url,inference_endpoint


def is_root_span(curr_span: Span) -> bool:
    return curr_span.parent is None


def get_input_from_args(chain_args):
    if len(chain_args) > 0 and isinstance(chain_args[0], str):
        return chain_args[0]
    return ""


def update_span_from_llm_response(response, span: Span, instance):
    # extract token uasge from langchain openai
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

    if isinstance(input_arg_text, dict):
        span.add_event(PROMPT_INPUT_KEY, input_arg_text)
    else:
        span.add_event(PROMPT_INPUT_KEY, {QUERY: input_arg_text})


def update_span_with_prompt_output(to_wrap, wrapped_args, span: Span):
    package_name: str = to_wrap.get('package')
    if isinstance(wrapped_args, str):
        span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: wrapped_args})
    if isinstance(wrapped_args, dict):
        span.add_event(PROMPT_OUTPUT_KEY, wrapped_args)
    if "llama_index.core.base.base_query_engine" in package_name:
        span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE: wrapped_args.response})
