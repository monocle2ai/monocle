

import logging
import os
import subprocess
import sys
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.retrievers.in_memory.embedding_retriever import (
    InMemoryDocumentStore,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup():
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

def test_haystack_sample(setup):
    from haystack.components.embedders import (
        SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder,
    )
    api_key = os.getenv("OPENAI_API_KEY")
    generator = OpenAIGenerator(
        api_key=Secret.from_token(api_key), model="gpt-4"
    )
    # api_key = os.getenv("MISTRAL_API_KEY")
    # generator = MistralChatGenerator(api_key=Secret.from_token(api_key), model="mistral-small")

    # initialize document store, load data and store in document store
    document_store = InMemoryDocumentStore()
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])

    # embedder to embed user query
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # get relevant documents from embedded query
    retriever = InMemoryEmbeddingRetriever(document_store)

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
    # template = [ChatMessage.from_user(template)]
    # prompt_builder = ChatPromptBuilder(template=template)

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

    logger.info(response["llm"]["replies"][0])
    time.sleep(10)
    spans = setup.get_captured_spans()

    for span in spans:
        span_attributes = span.attributes
        if span_attributes["span.type"] == "retrieval":
            # Assertions for all retrieval attributes
            assert span_attributes["entity.1.name"] == "InMemoryDocumentStore"
            assert span_attributes["entity.1.type"] == "vectorstore.InMemoryDocumentStore"
            assert span_attributes["entity.2.name"] == "sentence-transformers/all-MiniLM-L6-v2"
            assert span_attributes["entity.2.type"] == "model.embedding.sentence-transformers/all-MiniLM-L6-v2"

        if span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert span_attributes["entity.1.inference_endpoint"] == "https://api.openai.com/v1/"
            assert span_attributes["entity.2.name"] == "gpt-4"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if not span.parent and 'haystack' in span.name:  # Root span
            assert span_attributes["entity.1.name"] == "haystack_app_1"
            assert span_attributes["entity.1.type"] == "workflow.haystack"


# {
#     "name": "haystack.retriever",
#     "context": {
#         "trace_id": "0x627bf88dcb4903b36ec8e1981d974e30",
#         "span_id": "0x51147cff2fbf2ae1",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0b377cab5084badf",
#     "start_time": "2024-11-25T08:06:36.456448Z",
#     "end_time": "2024-11-25T08:06:36.469802Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "InMemoryDocumentStore",
#         "entity.1.type": "vectorstore.InMemoryDocumentStore",
#         "entity.2.name": "sentence-transformers/all-MiniLM-L6-v2",
#         "entity.2.type": "model.embedding.sentence-transformers/all-MiniLM-L6-v2"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-25T08:06:36.456477Z",
#             "attributes": {
#                 "input": "What does Rhodes Statue look like?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-25T08:06:36.469782Z",
#             "attributes": {
#                 "response": "Within it, too, are to be seen large masses of rock, by the weight of which the artist steadied it w..."
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
# },
# {
#     "name": "haystack.components.generators.openai.OpenAIGenerator",
#     "context": {
#         "trace_id": "0x627bf88dcb4903b36ec8e1981d974e30",
#         "span_id": "0xfca6cddc2de93669",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0b377cab5084badf",
#     "start_time": "2024-11-25T08:06:36.470300Z",
#     "end_time": "2024-11-25T08:06:38.720635Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_openai",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-3.5-turbo",
#         "entity.2.type": "model.llm.gpt-3.5-turbo"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-25T08:06:38.720289Z",
#             "attributes": {
#                 "input": "What does Rhodes Statue look like?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-25T08:06:38.720316Z",
#             "attributes": {
#                 "response": "The Rhodes Statue was a giant statue of the Greek sun-god Helios, erected in the city of Rhodes. It stood approximately 33 meters (108 feet) tall and was made of bronze. Scholars do not know the exact appearance of the statue, but it is believed to have had curly hair with evenly spaced spikes of bronze or silver flame radiating from the head, similar to contemporary Rhodian coins. The statue was located near the Rhodes harbor entrance on a 15-meter-high white marble pedestal. It collapsed at the knees during an earthquake in 226 BC."
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2024-11-25T08:06:38.720610Z",
#             "attributes": {
#                 "completion_tokens": 114,
#                 "prompt_tokens": 2464,
#                 "total_tokens": 2578
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
#         "trace_id": "0x627bf88dcb4903b36ec8e1981d974e30",
#         "span_id": "0x0b377cab5084badf",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-25T08:06:36.445584Z",
#     "end_time": "2024-11-25T08:06:38.720920Z",
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
#             "timestamp": "2024-11-25T08:06:36.446344Z",
#             "attributes": {
#                 "input": "What does Rhodes Statue look like?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-25T08:06:38.720906Z",
#             "attributes": {
#                 "response": [
#                     "The Rhodes Statue was a giant statue of the Greek sun-god Helios, erected in the city of Rhodes. It stood approximately 33 meters (108 feet) tall and was made of bronze. Scholars do not know the exact appearance of the statue, but it is believed to have had curly hair with evenly spaced spikes of bronze or silver flame radiating from the head, similar to contemporary Rhodian coins. The statue was located near the Rhodes harbor entrance on a 15-meter-high white marble pedestal. It collapsed at the knees during an earthquake in 226 BC."
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
