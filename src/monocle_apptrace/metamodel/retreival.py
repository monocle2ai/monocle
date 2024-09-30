from attr import attributes
from monocle_apptrace.utils import resolve_from_alias, update_span_with_infra_name, with_tracer_wrapper,get_embedding_model
from monocle_apptrace.metamodel.embeded_model import EmbedModel
from monocle_apptrace.metamodel.vector_store import VectorStore
from msrest.serialization import attribute_transformer


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
class Retreival:
    def __init__(self, instance):
        self.instance = instance


    def add_attributes(self, span,to_wrap):
        #Resolve model_name using the alias list
        package = to_wrap.get('package')
        attributes=''
        embedding_model=''
        if package in framework_vector_store_mapping:
            attributes = framework_vector_store_mapping[package](self.instance)
            embedding_model=attributes['embedding_model']
            provider = attributes['provider']
            span.set_attribute("span.type", "Retreival")
            span.set_attribute("entity.count", 2)
            vectorstore = VectorStore(self.instance)
            embedmodel = EmbedModel(self.instance)

            vectorstore.add_vectorstore_attributes(span,provider, embedding_model)
            embedmodel.add_embedmodel_attributes(span, embedding_model)
