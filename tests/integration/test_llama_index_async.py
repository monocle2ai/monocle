

import os
import asyncio
import time
import chromadb
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI, AsyncAzureOpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from tests.common.helpers import find_span_by_type, find_spans_by_type, validate_inference_span_events, verify_inference_span

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="llama_index_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

@pytest.mark.integration()
def test_llama_index_sample(setup: None):    
    # Creating a Chroma client
    # EphemeralClient operates purely in-memory, PersistentClient will also save to disk
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart-async")

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

        model="gpt-4o-mini",)
    # llm = MistralAI(api_key=os.getenv("MISTRAL_API_KEY"))

    query_engine = index.as_query_engine(llm= llm, )
    response = asyncio.run(query_engine.aquery("What did the author do growing up?"))
    time.sleep(5)  # Allow time for spans to be captured
    spans = custom_exporter.get_captured_spans()

    assert len(spans) > 0, "No spans captured for the LangChain Anthropic sample"
    retrival_span = None
    for span in spans:
        span_attributes = span.attributes
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "retrieval"
        ):
            # Assertions for all retrieval attributes
            assert span_attributes["entity.1.name"] == "ChromaVectorStore"
            assert span_attributes["entity.1.type"] == "vectorstore.ChromaVectorStore"
            assert span_attributes["entity.2.name"] == "text-embedding-3-large"
            assert (
                span_attributes["entity.2.type"]
                == "model.embedding.text-embedding-3-large"
            )
            assert not span.name.lower().startswith("openai")
            retrival_span = span
    assert retrival_span is not None, "Expected to find retrieval span"
    workflow_span = None

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        # Also check for inference.framework spans
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify each inference span
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.azure_openai",
            model_name="gpt-4o-mini",
            model_type="model.llm.gpt-4o-mini",
            check_metadata=True,
            check_input_output=True,
        )
    assert (
        len(inference_spans) == 1
    ), "Expected exactly one inference span for the LLM call"

    # Validate events using the generic function with regex patterns
    validate_inference_span_events(
        span=inference_spans[0],
        expected_event_count=3,
        input_patterns=[
            r"^\{\"system': \".+\"\}$",  # Pattern for system message
            r"^\{\"user\": \".+\"\}$",  # Pattern for user message
        ],
        output_pattern=r"^\{'assistant': '.+'\}$",  # Pattern for assistant response
        metadata_requirements={
            "temperature": float,
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int,
        },
    )
    # pick the last span as there is are two workflow spans:
    # one for openai embedding one for llamaindex query
    workflow_span = find_span_by_type([spans[-1]], "workflow")

    assert workflow_span is not None, "Expected to find workflow span"

    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "llama_index_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.llamaindex"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

    

# {
#     "name": "openai.resources.embeddings.Embeddings",
#     "context": {
#         "trace_id": "0x8f46b23ce45e4c7ab19247ce936b84ba",
#         "span_id": "0x56a21d58e0e1306c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9c930de804e361ad",
#     "start_time": "2025-07-02T20:48:22.203605Z",
#     "end_time": "2025-07-02T20:48:23.032892Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/llama_index/embeddings/openai/base.py:169",
#         "workflow.name": "llama_index_1",
#         "span.type": "embedding",
#         "entity.1.name": "text-embedding-3-large",
#         "entity.1.type": "model.embedding.text-embedding-3-large",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-02T20:48:23.030698Z",
#             "attributes": {
#                 "input": "file_path: /Users/kshitizvijayvargiya/monocle-ksh/tests/integration/../data/sample.txt  this is some sample text"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T20:48:23.032821Z",
#             "attributes": {
#                 "response": "index=0, embedding=[0.010904251597821712, 0.006519173737615347, -0.006850976962596178, 0.03867012634..."
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
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x8f46b23ce45e4c7ab19247ce936b84ba",
#         "span_id": "0x9c930de804e361ad",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T20:48:22.203550Z",
#     "end_time": "2025-07-02T20:48:23.032982Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/llama_index/embeddings/openai/base.py:169",
#         "span.type": "workflow",
#         "entity.1.name": "llama_index_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.embeddings.AsyncEmbeddings",
#     "context": {
#         "trace_id": "0x13e49d8ff3e6744795abcc5b05146143",
#         "span_id": "0xb3dcd25acabdfe18",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf3546fbcefba9f83",
#     "start_time": "2025-07-02T20:48:23.066565Z",
#     "end_time": "2025-07-02T20:48:23.666874Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/llama_index/embeddings/openai/base.py:147",
#         "workflow.name": "llama_index_1",
#         "span.type": "embedding.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "llama_index.core.indices.vector_store.retrievers.retriever.VectorIndexRetriever",
#     "context": {
#         "trace_id": "0x13e49d8ff3e6744795abcc5b05146143",
#         "span_id": "0xf3546fbcefba9f83",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xded6ee5d79ebd5d6",
#     "start_time": "2025-07-02T20:48:23.057045Z",
#     "end_time": "2025-07-02T20:48:23.676253Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/llama_index/core/query_engine/retriever_query_engine.py:141",
#         "workflow.name": "llama_index_1",
#         "span.type": "retrieval",
#         "entity.1.name": "ChromaVectorStore",
#         "entity.1.type": "vectorstore.ChromaVectorStore",
#         "entity.2.name": "text-embedding-3-large",
#         "entity.2.type": "model.embedding.text-embedding-3-large",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-02T20:48:23.676206Z",
#             "attributes": {
#                 "input": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T20:48:23.676233Z",
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
#     "name": "openai.resources.chat.completions.completions.AsyncCompletions",
#     "context": {
#         "trace_id": "0x13e49d8ff3e6744795abcc5b05146143",
#         "span_id": "0x38854e96ae93a682",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x71ee0311cb5bb852",
#     "start_time": "2025-07-02T20:48:23.693076Z",
#     "end_time": "2025-07-02T20:48:24.822180Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/llama_index/llms/openai/base.py:743",
#         "workflow.name": "llama_index_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "llama_index.llms.azure_openai.base.AzureOpenAI",
#     "context": {
#         "trace_id": "0x13e49d8ff3e6744795abcc5b05146143",
#         "span_id": "0x71ee0311cb5bb852",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xded6ee5d79ebd5d6",
#     "start_time": "2025-07-02T20:48:23.681665Z",
#     "end_time": "2025-07-02T20:48:24.823654Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/llama_index/core/llms/llm.py:734",
#         "workflow.name": "llama_index_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.azure_openai",
#         "entity.1.deployment": "kshitiz-gpt",
#         "entity.1.inference_endpoint": "https://okahu-openai-dev.openai.azure.com/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-02T20:48:24.823536Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are an expert Q&A system that is trusted around the world.\\nAlways answer the query using the provided context information, and not prior knowledge.\\nSome rules to follow:\\n1. Never directly reference the given context in your answer.\\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\"}",
#                     "{\"user\": \"What did the author do growing up?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T20:48:24.823603Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"The author grew up reading and writing.\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T20:48:24.823630Z",
#             "attributes": {
#                 "temperature": 0.1,
#                 "completion_tokens": 8,
#                 "prompt_tokens": 152,
#                 "total_tokens": 160
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
#     "name": "llama_index.core.query_engine.retriever_query_engine.RetrieverQueryEngine",
#     "context": {
#         "trace_id": "0x13e49d8ff3e6744795abcc5b05146143",
#         "span_id": "0xded6ee5d79ebd5d6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe97ff8ec80b2924d",
#     "start_time": "2025-07-02T20:48:23.053762Z",
#     "end_time": "2025-07-02T20:48:24.823947Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_llama_index_async.py:65",
#         "workflow.name": "llama_index_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x13e49d8ff3e6744795abcc5b05146143",
#         "span_id": "0xe97ff8ec80b2924d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T20:48:23.053687Z",
#     "end_time": "2025-07-02T20:48:24.823967Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_llama_index_async.py:65",
#         "span.type": "workflow",
#         "entity.1.name": "llama_index_1",
#         "entity.1.type": "workflow.llamaindex",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
