

import os

import chromadb
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="llama_index_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

@pytest.mark.integration()
def test_llama_index_sample(setup):    
    # Creating a Chroma client
    # EphemeralClient operates purely in-memory, PersistentClient will also save to disk
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")

    # construct vector store
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection,
    )
    dir_path = os.path.dirname(os.path.realpath(__file__))
    documents = SimpleDirectoryReader(os.path.join(dir_path, "..", "data")).load_data()

    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    # llm = OpenAI(temperature=0.8, model="gpt-4")
    llm = AzureOpenAI(
        engine=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        temperature=0.1,
        # model="gpt-4",

        model="gpt-3.5-turbo-0125")

    # llm = MistralAI(api_key=os.getenv("MISTRAL_API_KEY"))

    query_engine = index.as_query_engine(llm= llm, )
    response = query_engine.query("What did the author do growing up?")

    print(response)

    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if "span.type" in span_attributes and span_attributes["span.type"] == "retrieval":
            # Assertions for all retrieval attributes
            assert span_attributes["entity.1.name"] == "ChromaVectorStore"
            assert span_attributes["entity.1.type"] == "vectorstore.ChromaVectorStore"
            assert span_attributes["entity.2.name"] == "text-embedding-3-large"
            assert span_attributes["entity.2.type"] == "model.embedding.text-embedding-3-large"

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.azure_oai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-3.5-turbo-0125"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-3.5-turbo-0125"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if not span.parent and span.name == "llamaindex.query":  # Root span
            assert span_attributes["entity.1.name"] == "llama_index_1"
            assert span_attributes["entity.1.type"] == "workflow.langchain"

# {
#     "name": "llamaindex.retrieve",
#     "context": {
#         "trace_id": "0x3d7d2ae55d97a559242748747f4a43e6",
#         "span_id": "0x275af74b6830184d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3ed7bbf9219a0a04",
#     "start_time": "2024-11-12T11:28:35.243346Z",
#     "end_time": "2024-11-12T11:28:36.080680Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "ChromaVectorStore",
#         "entity.1.type": "vectorstore.ChromaVectorStore",
#         "entity.2.name": "text-embedding-3-large",
#         "entity.2.type": "model.embedding.text-embedding-3-large"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-12T11:28:35.243346Z",
#             "attributes": {
#                 "question": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-12T11:28:36.080680Z",
#             "attributes": {
#                 "response": "this is some sample text"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "llamaindex.openai",
#     "context": {
#         "trace_id": "0x3d7d2ae55d97a559242748747f4a43e6",
#         "span_id": "0xde43691c9c4aa1c7",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3ed7bbf9219a0a04",
#     "start_time": "2024-11-12T11:28:36.082708Z",
#     "end_time": "2024-11-12T11:28:37.999529Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1",
#         "entity.2.name": "gpt-4",
#         "entity.2.type": "model.llm.gpt-4"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-18T10:57:06.165465Z",
#             "attributes": {
#                 "system": "You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.",
#                 "user": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-18T10:57:06.165494Z",
#             "attributes": {
#                 "assistant": "The context does not provide information about what the author did while growing up."
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2024-11-12T11:28:37.999529Z",
#             "attributes": {
#                 "temperature": 0.1,
#                 "completion_tokens": 15,
#                 "prompt_tokens": 149,
#                 "total_tokens": 164
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "llamaindex.query",
#     "context": {
#         "trace_id": "0x3d7d2ae55d97a559242748747f4a43e6",
#         "span_id": "0x3ed7bbf9219a0a04",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-12T11:28:35.242345Z",
#     "end_time": "2024-11-12T11:28:37.999529Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "workflow",
#         "entity.1.name": "llama_index_1",
#         "entity.1.type": "workflow.llamaindex",
#         "monocle_apptrace.version": "0.2.0"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-21T10:36:11.614035Z",
#             "attributes": {
#                 "input": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-21T10:36:14.461733Z",
#             "attributes": {
#                 "response": "There is no information provided in the context about what the author did growing up."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }