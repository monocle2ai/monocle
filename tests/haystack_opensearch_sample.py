import os
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.retrievers.opensearch  import OpenSearchEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from monocle_apptrace.wrap_common import llm_wrapper, task_wrapper
from monocle_apptrace.wrapper import WrapperMethod

def haystack_app():

    setup_monocle_telemetry(
            workflow_name="haystack_app_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[


            ])

    # initialize

    api_key = os.getenv("OPENAI_API_KEY")
    http_auth=("sachin-opensearch", "Sachin@123")
    generator = OpenAIGenerator(
        api_key=Secret.from_token(api_key), model="gpt-3.5-turbo"
    )
    document_store = OpenSearchDocumentStore(hosts="https://search-sachin-opensearch-cvvd5pdeyrme2l2y26xmcpkm2a.us-east-1.es.amazonaws.com", use_ssl=True,
                        verify_certs=True, http_auth=http_auth)
    model = "sentence-transformers/all-mpnet-base-v2"

    # documents = [Document(content="There are over 7,000 languages spoken around the world today."),
    #                         Document(content="Elephants have been observed to behave in a way that indicates a high level of self-awareness, such as recognizing themselves in mirrors."),
    #                         Document(content="In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness the phenomenon of bioluminescent waves.")]

    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
    document_embedder = SentenceTransformersDocumentEmbedder(model=model)
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)

    document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.SKIP)


    # embedder to embed user query
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-mpnet-base-v2"
    )

    # get relevant documents from embedded query
    retriever = OpenSearchEmbeddingRetriever(document_store=document_store)

    # use documents to build the prompt
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)

    basic_rag_pipeline = Pipeline()
    # Add components to your pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)

    # Now, connect the components to each other
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    question = "What does Rhodes Statue look like?"

    response = basic_rag_pipeline.run(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
    )

    # print(response["llm"]["replies"][0])


haystack_app()

# {
#     "name": "haystack.retriever",
#     "context": {
#         "trace_id": "0xbe2f5ba695532ee254540d231824080d",
#         "span_id": "0x53d1f82b8a65f332",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xfdb6b305da0ea91d",
#     "start_time": "2024-11-19T12:16:20.688720Z",
#     "end_time": "2024-11-19T12:16:21.461968Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "OpenSearchDocumentStore",
#         "entity.1.type": "vectorstore.OpenSearchDocumentStore",
#         "entity.1.deployment": "https://search-sachin-opensearch-cvvd5pdeyrme2l2y26xmcpkm2a.us-east-1.es.amazonaws.com:443",
#         "entity.2.name": "sentence-transformers/all-mpnet-base-v2",
#         "entity.2.type": "model.embedding.sentence-transformers/all-mpnet-base-v2"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "haystack.components.generators.openai.OpenAIGenerator",
#     "context": {
#         "trace_id": "0xbe2f5ba695532ee254540d231824080d",
#         "span_id": "0xc40c559c2fbbd32b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xfdb6b305da0ea91d",
#     "start_time": "2024-11-19T12:16:21.461968Z",
#     "end_time": "2024-11-19T12:16:23.486734Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo",
#         "entity.2.type": "model.llm.gpt-3.5-turbo"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-11-19T12:16:23.486734Z",
#             "attributes": {
#                 "completion_tokens": 120,
#                 "prompt_tokens": 2433,
#                 "total_tokens": 2553
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "haystack.core.pipeline.pipeline.Pipeline",
#     "context": {
#         "trace_id": "0xbe2f5ba695532ee254540d231824080d",
#         "span_id": "0xfdb6b305da0ea91d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-19T12:16:20.621638Z",
#     "end_time": "2024-11-19T12:16:23.487739Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "entity.1.name": "haystack_app_1",
#         "entity.1.type": "workflow.haystack"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-19T12:16:20.621638Z",
#             "attributes": {
#                 "question": "What does Rhodes Statue look like?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-19T12:16:23.487739Z",
#             "attributes": {
#                 "response": [
#                     "The Rhodes Statue, also known as the Colossus of Rhodes, was a 33-meter (108 feet) tall statue of the Greek sun-god Helios. It was constructed with iron tie bars covered in brass plates to form the skin, and filled with stone blocks. The head of the statue had curly hair with spikes of bronze or silver flame radiating. The statue also had a gilded robe made from glass, carved with animals and lilies, and held a small chryselephantine statue of crowned Nike in its right hand and a scepter in its left hand."
#                 ]
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }