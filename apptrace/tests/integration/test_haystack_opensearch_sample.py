import subprocess
import logging
import os
import sys
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.document_stores.opensearch.auth import AWSAuth
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opensearchpy.helpers import bulk

logger = logging.getLogger(__name__)


def write_documents_serverless(document_store, documents):
    """Ingest documents into an AWS OpenSearch Serverless (aoss) collection.

    Uses opensearch-haystack private attributes (`_client`, `_index`,
    `_max_chunk_bytes`); there is no public accessor for them in 8.x.
    """
    # Also creates the index with its knn mapping if it does not exist yet.
    document_store._ensure_initialized()

    # Without `_id` the collection cannot de-duplicate, so an unconditional
    # write would append the whole dataset again on every run.
    already_present = document_store.count_documents()
    if already_present > 0:
        logger.info("OpenSearch index already holds %s documents, skipping ingest", already_present)
        return 0

    actions = []
    for doc in documents:
        source = doc.to_dict()
        # Mirrors OpenSearchDocumentStore: sparse embeddings are not storable.
        source.pop("sparse_embedding", None)
        actions.append({"_op_type": "index", "_source": source})

    # No `refresh`: serverless collections do not support it.
    written, errors = bulk(
        client=document_store._client,
        actions=actions,
        index=document_store._index,
        raise_on_error=False,
        max_chunk_bytes=document_store._max_chunk_bytes,
    )
    if errors:
        raise RuntimeError(f"Failed to write documents to OpenSearch: {errors}")

    # Serverless indexing is eventually consistent, so without `refresh` the
    # retriever can otherwise run before any of these documents are visible.
    for _ in range(30):
        if document_store.count_documents() >= written:
            break
        time.sleep(2)
    else:
        logger.warning("Ingested documents not yet searchable; retrieval may return nothing")
    return written


@pytest.fixture(scope="module")
def setup():
    instrumentor = None
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", ".[dev_tranformers]"])
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="haystack_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[
            ])
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

@pytest.fixture(scope="module", autouse=True)
def cleanup_module():
    yield
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y","sentence-transformers"])

def test_haystack_opensearch_sample(setup):
    from haystack_integrations.components.embedders.sentence_transformers import (
        SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder,
    )
    # initialize
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
    aws_auth = AWSAuth(aws_access_key_id=Secret.from_token(os.environ["AWS_ACCESS_KEY_ID"]),
                        aws_secret_access_key=Secret.from_token(os.environ["AWS_SECRET_ACCESS_KEY"]),
                        aws_region_name=Secret.from_token(os.environ["AWS_DEFAULT_REGION"]),
                        aws_service='aoss')
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(api_key), model="gpt-4"
    )
    document_store = OpenSearchDocumentStore(hosts=os.getenv("OPEN_SEARCH_DOCSTORE_ENDPOINT"), use_ssl=True,
                        verify_certs=True, http_auth=aws_auth)

    # documents = [Document(content="There are over 7,000 languages spoken around the world today."),
    #                         Document(content="Elephants have been observed to behave in a way that indicates a high level of self-awareness, such as recognizing themselves in mirrors."),
    #                         Document(content="In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness the phenomenon of bioluminescent waves.")]

    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
    document_embedder = SentenceTransformersDocumentEmbedder()
    documents_with_embeddings = document_embedder.run(documents)

    write_documents_serverless(document_store,documents_with_embeddings.get("documents"))


    # embedder to embed user query
    text_embedder = SentenceTransformersTextEmbedder()

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

    # logger.info(response["llm"]["replies"][0])

    spans = setup.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if "span.type" in span_attributes and span_attributes["span.type"] == "retrieval":
            # Assertions for all retrieval attributes
            assert span_attributes["entity.1.name"] == "OpenSearchDocumentStore"
            assert span_attributes["entity.1.type"] == "vectorstore.OpenSearchDocumentStore"
            assert "entity.1.deployment" in span_attributes
            assert span_attributes["entity.2.name"] == "sentence-transformers/all-mpnet-base-v2"
            assert span_attributes["entity.2.type"] == "model.embedding.sentence-transformers/all-mpnet-base-v2"

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.azure_openai"
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4"

        if not span.parent and 'haystack' in span.name:  # Root span
            assert span_attributes["entity.1.name"] == "haystack_app_1"
            assert span_attributes["entity.1.type"] == "workflow.haystack"

# {
#     "name": "haystack.retriever",
#     "context": {
#         "trace_id": "0xa599cf84e013b83c58e3afaf8a7058f8",
#         "span_id": "0x90b01a17810b9b38",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x557fc857283d8651",
#     "start_time": "2024-11-26T09:52:00.845732Z",
#     "end_time": "2024-11-26T09:52:01.742785Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "OpenSearchDocumentStore",
#         "entity.1.type": "vectorstore.OpenSearchDocumentStore",
#         "entity.1.deployment": "https://xyz.us-east-1.es.amazonaws.com:443",
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
#     "name": "haystack.components.generators.openai.OpenAIChatGenerator",
#     "context": {
#         "trace_id": "0xa599cf84e013b83c58e3afaf8a7058f8",
#         "span_id": "0x1de03fa69ab19977",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x557fc857283d8651",
#     "start_time": "2024-11-26T09:52:01.742785Z",
#     "end_time": "2024-11-26T09:52:03.804858Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_openai",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-11-26T09:52:03.804858Z",
#             "attributes": {
#                 "completion_tokens": 126,
#                 "prompt_tokens": 2433,
#                 "total_tokens": 2559
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
#         "trace_id": "0xa599cf84e013b83c58e3afaf8a7058f8",
#         "span_id": "0x557fc857283d8651",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-26T09:52:00.681588Z",
#     "end_time": "2024-11-26T09:52:03.805858Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.2.0",
#         "span.type": "workflow",
#         "entity.1.name": "haystack_app_1",
#         "entity.1.type": "workflow.haystack"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-26T09:52:00.684591Z",
#             "attributes": {
#                 "question": "What does Rhodes Statue look like?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-26T09:52:03.805858Z",
#             "attributes": {
#                 "response": [
#                     "The Rhodes Statue was a colossal statue of the Greek sun-god Helios, standing approximately 33 meters (108 feet) high. It featured a standard rendering of a head with curly hair and spikes of bronze or silver flame radiating from it. The statue was constructed with iron tie bars and brass plates to form the skin, and filled with stone blocks during construction. The statue collapsed at the knees during an earthquake in 226 BC and remained on the ground for over 800 years. It was ultimately destroyed and the remains were sold. The exact appearance of the statue, aside from its size and head details, is unknown."
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
