#pylint: disable=protected-access
import logging
import os
import inspect
from urllib.parse import urlparse
from opentelemetry.trace import Span, Tracer
from monocle_apptrace.utils import resolve_from_alias, update_span_with_infra_name, with_tracer_wrapper, get_embedding_model, get_context_attribute

logger = logging.getLogger(__name__)
WORKFLOW_TYPE_KEY = "workflow_type"
CONTEXT_INPUT_KEY = "context_input"
CONTEXT_OUTPUT_KEY = "context_output"
PROMPT_INPUT_KEY = "input"
PROMPT_OUTPUT_KEY = "output"
QUERY = "question"
RESPONSE = "response"
TAGS = "tags"
SESSION_PROPERTIES_KEY = "session"
INFRA_SERVICE_KEY = "infra_service_name"
TYPE = "type"
PROVIDER = "provider_name"
EMBEDDING_MODEL = "embedding_model"
VECTOR_STORE = 'vector_store'


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
        pre_task_processing(to_wrap, instance, args, span)
        return_value = wrapped(*args, **kwargs)
        post_task_processing(to_wrap, span, return_value)

    return return_value

def post_task_processing(to_wrap, span, return_value):
    update_span_with_context_output(to_wrap=to_wrap, return_value=return_value, span=span)

    if is_root_span(span):
        workflow_name = span.resource.attributes.get("service.name")
        span.set_attribute("workflow_name",workflow_name)
        update_span_with_prompt_output(to_wrap=to_wrap, wrapped_args=return_value, span=span)
        update_workflow_type(to_wrap, span)

def pre_task_processing(to_wrap, instance, args, span):
    if is_root_span(span):
        update_span_with_prompt_input(to_wrap=to_wrap, wrapped_args=args, span=span)

        update_span_with_infra_name(span, INFRA_SERVICE_KEY)

    #capture the tags attribute of the instance if present, else ignore
    try:
        update_tags(to_wrap, instance, span)
        update_vectorstore_attributes(to_wrap, instance, span)
    except AttributeError:
        pass
    update_span_with_context_input(to_wrap=to_wrap, wrapped_args=args, span=span)



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
        update_llm_endpoint(curr_span= span, instance=instance)

        return_value = await wrapped(*args, **kwargs)
        update_span_from_llm_response(response = return_value, span = span)

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
            update_tags(to_wrap, instance, span)
            update_vectorstore_attributes(to_wrap, instance, span)
            input_arg_text = get_context_attribute(CONTEXT_INPUT_KEY)
            span.add_event(CONTEXT_INPUT_KEY, {QUERY: input_arg_text})
        update_llm_endpoint(curr_span= span, instance=instance)

        return_value = wrapped(*args, **kwargs)
        if 'haystack.components.retrievers' in to_wrap['package'] and 'haystack.retriever' in span.name:
            update_span_with_context_output(to_wrap=to_wrap, return_value=return_value, span=span)
        update_span_from_llm_response(response = return_value, span = span)

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

        set_provider_name(curr_span, instance)

def set_provider_name(curr_span, instance):
    provider_url = ""

    try :
        if isinstance(instance.client._client.base_url.host, str) :
            provider_url = instance. client._client.base_url.host
    except:
        pass

    try :
        if isinstance(instance.api_base, str):
            provider_url = instance.api_base
    except:
        pass

    try :
        if len(provider_url) > 0:
            parsed_provider_url = urlparse(provider_url)
            curr_span.set_attribute("provider_name", parsed_provider_url.hostname or provider_url)
    except:
        pass

def is_root_span(curr_span: Span) -> bool:
    return curr_span.parent is None

def get_input_from_args(chain_args):
    if len(chain_args) > 0 and isinstance(chain_args[0], str):
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
        try:
            if response.raw is not None:
                token_usage = response.raw.get("usage") if isinstance(response.raw, dict) else getattr(response.raw, "usage", None)
                if token_usage is not None:
                    if getattr(token_usage, "completion_tokens", None):
                        span.set_attribute("completion_tokens", getattr(token_usage, "completion_tokens"))
                    if getattr(token_usage, "prompt_tokens", None):
                        span.set_attribute("prompt_tokens", getattr(token_usage, "prompt_tokens"))
                    if getattr(token_usage, "total_tokens", None):
                        span.set_attribute("total_tokens", getattr(token_usage, "total_tokens"))
        except AttributeError:
            token_usage = None


def update_workflow_type(to_wrap, span: Span):
    package_name = to_wrap.get('package')

    for (package, workflow_type) in WORKFLOW_TYPE_MAP.items():
        if(package_name is not None and package in package_name):
            span.set_attribute(WORKFLOW_TYPE_KEY, workflow_type)

def update_span_with_context_input(to_wrap, wrapped_args ,span: Span):
    package_name: str = to_wrap.get('package')
    if "langchain_core.retrievers" in package_name:
        input_arg_text = wrapped_args[0]
        span.add_event(CONTEXT_INPUT_KEY, {QUERY:input_arg_text})
    if "llama_index.core.indices.base_retriever" in package_name:
        input_arg_text = wrapped_args[0].query_str
        span.add_event(CONTEXT_INPUT_KEY, {QUERY:input_arg_text})
    if "haystack.components.retrievers.in_memory" in package_name:
        input_arg_text = get_context_attribute(CONTEXT_INPUT_KEY)
        span.add_event(CONTEXT_INPUT_KEY, {QUERY:input_arg_text})

def update_span_with_context_output(to_wrap, return_value ,span: Span):
    package_name: str = to_wrap.get('package')
    if "langchain_core.retrievers" in package_name:
        combined_output = " ".join([doc.page_content for doc in return_value if hasattr(doc, 'page_content')])
        if len(combined_output) > 100:
            combined_output = combined_output[:100] + "..."

        span.add_event(CONTEXT_OUTPUT_KEY, {RESPONSE: combined_output})
    if "llama_index.core.indices.base_retriever" in package_name:
        output_arg_text = return_value[0].text
        span.add_event(CONTEXT_OUTPUT_KEY, {RESPONSE:output_arg_text})

    if "haystack.components.retrievers.in_memory" in package_name:
        combined_output = " ".join([doc.content for doc in return_value['documents']])
        if len(combined_output) > 100:
            combined_output = combined_output[:100] + "..."
        span.add_event(CONTEXT_OUTPUT_KEY, {RESPONSE: combined_output})

def update_span_with_prompt_input(to_wrap, wrapped_args ,span: Span):
    input_arg_text = wrapped_args[0]

    if isinstance(input_arg_text, dict):
        span.add_event(PROMPT_INPUT_KEY,input_arg_text)
    else:
        span.add_event(PROMPT_INPUT_KEY,{QUERY:input_arg_text})

def update_span_with_prompt_output(to_wrap, wrapped_args ,span: Span):
    package_name: str = to_wrap.get('package')
    if isinstance(wrapped_args, str):
        span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE:wrapped_args})
    if "llama_index.core.base.base_query_engine" in package_name:
        span.add_event(PROMPT_OUTPUT_KEY, {RESPONSE:wrapped_args.response})

def update_tags(to_wrap, instance, span):
    try:
        # copy tags as is from langchain
        if hasattr(instance, TAGS):
            tags_value = getattr(instance, TAGS)
            if tags_value is not None:
                span.set_attribute(TAGS, getattr(instance, TAGS))
    except:
        pass
    try:
        # extract embed model and vector store names for llamaindex
        package_name: str = to_wrap.get('package')
        if "llama_index.core.indices.base_retriever" in package_name:
            model_name = instance._embed_model.__class__.__name__
            vector_store_name = type(instance._vector_store).__name__
            span.set_attribute(TAGS, [model_name, vector_store_name])
        if "haystack.components.retrievers.in_memory" in package_name:
            model_name = instance.__dict__.get('__haystack_added_to_pipeline__').get_component('text_embedder').__class__.__name__
            vector_store_name = instance.__dict__.get("document_store").__class__.__name__
            span.set_attribute(TAGS, [model_name, vector_store_name])
    except:
        pass


def update_vectorstore_attributes(to_wrap, instance, span):
    """
       Updates the telemetry span attributes for vector store retrieval tasks.
    """
    try:
        package = to_wrap.get('package')
        if package in framework_vector_store_mapping:
            attributes = framework_vector_store_mapping[package](instance)
            span._attributes.update({
                TYPE: attributes['type'],
                PROVIDER: attributes['provider'],
                EMBEDDING_MODEL: attributes['embedding_model']
            })
        else:
            pass

    except Exception as e:
        logger.error(f"Error updating span attributes: {e}")
