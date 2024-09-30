from monocle_apptrace.utils import resolve_from_alias, update_span_with_infra_name, with_tracer_wrapper
from monocle_apptrace.utils import get_embedding_model
from virtualenv.config.convert import NoneType

VECTOR_STORE = 'vector_store'
framework_vector_store_mapping = {
        'langchain_core.retrievers': lambda instance: {
            'provider': instance.tags[0],
            'embedding_model': instance.tags[1],
            'type': VECTOR_STORE,
        },
        'llama_index.core.indices.base_retriever': lambda instance: {
            'provider': type(instance._vector_store).__name__,
            'embedding_model': instance._embed_model.model_name,
            'type': VECTOR_STORE,
        },
        'haystack.components.retrievers': lambda instance: {
            'provider': instance.__dict__.get("document_store").__class__.__name__,
            'embedding_model': get_embedding_model(),
            'type': VECTOR_STORE,
        },
    }
class VectorStore:
    def __init__(self, instance):
        self.instance = instance

    def add_vectorstore_attributes(self, span,provider,embedding_model):
        # Resolve model_name using the alias list
        #vector_store = self.instance.__dict__.get('_vector_store').__class__.__name__

        # Check if _vector_store exists and get its class name

        span.set_attribute("entity.1.name", provider)
        span.set_attribute("entity.1.type", "vectorstore.chroma")
        span.set_attribute("entity.1.embedding_model_name", embedding_model)
