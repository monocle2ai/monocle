

import os

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from monocle_apptrace.wrap_common import llm_wrapper
from monocle_apptrace.wrapper import WrapperMethod
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

setup_monocle_telemetry(
    workflow_name="llama_index_1",
    span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
    wrapper_methods=[]
)

# Creating a Chroma client
# EphemeralClient operates purely in-memory, PersistentClient will also save to disk
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)
dir_path = os.path.dirname(os.path.realpath(__file__))
documents = SimpleDirectoryReader(dir_path + "/data").load_data()

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

llm = OpenAI(temperature=0.1, model="gpt-4")

query_engine = index.as_query_engine(llm= llm, )
response = query_engine.query("What did the author do growing up?")

print(response)

# {
#     "name": "haystack.tracing.auto_enable",
#     "context": {
#         "trace_id": "0xf199d425ff9455d2fa18da30508e8120",
#         "span_id": "0xced2da35b439e06c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-13T11:37:37.143875Z",
#     "end_time": "2024-09-13T11:37:37.144983Z",
#     "status": {
#         "status_code": "ERROR",
#         "description": "ImportError: cannot import name 'Span' from partially initialized module 'haystack.tracing' (most likely due to a circular import) (/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/__init__.py)"
#     },
#     "attributes": {},
#     "events": [
#         {
#             "name": "exception",
#             "timestamp": "2024-09-13T11:37:37.144968Z",
#             "attributes": {
#                 "exception.type": "ImportError",
#                 "exception.message": "cannot import name 'Span' from partially initialized module 'haystack.tracing' (most likely due to a circular import) (/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/__init__.py)",
#                 "exception.stacktrace": "Traceback (most recent call last):\n  File \"/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/opentelemetry/trace/__init__.py\", line 590, in use_span\n    yield span\n  File \"/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/opentelemetry/sdk/trace/__init__.py\", line 1108, in start_as_current_span\n    yield span\n  File \"/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/tracer.py\", line 207, in _auto_configured_opentelemetry_tracer\n    from haystack.tracing.opentelemetry import OpenTelemetryTracer\n  File \"/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/opentelemetry.py\", line 9, in <module>\n    from haystack.tracing import Span, Tracer\nImportError: cannot import name 'Span' from partially initialized module 'haystack.tracing' (most likely due to a circular import) (/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/__init__.py)\n",
#                 "exception.escaped": "False"
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
#     "name": "haystack.tracing.auto_enable",
#     "context": {
#         "trace_id": "0x9be2be7e2994b5f923d17defaed1c00e",
#         "span_id": "0xc790b64df4a91ff6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-13T11:37:38.692198Z",
#     "end_time": "2024-09-13T11:37:38.692245Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
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
#     "name": "llamaindex.retrieve",
#     "context": {
#         "trace_id": "0x79de813d316bd63f40767fec67560d0f",
#         "span_id": "0xd9661af547dfcfc1",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa9e16efccceb9368",
#     "start_time": "2024-09-13T11:37:40.848528Z",
#     "end_time": "2024-09-13T11:37:41.760007Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-09-13T11:37:40.848554Z",
#             "attributes": {
#                 "question": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "context_output",
#             "timestamp": "2024-09-13T11:37:41.759986Z",
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
# The context does not provide information about what the author did while growing up.
# {
#     "name": "llamaindex.openai",
#     "context": {
#         "trace_id": "0x79de813d316bd63f40767fec67560d0f",
#         "span_id": "0xf17975e22cc1f920",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x99c6a02cfdfddb24",
#     "start_time": "2024-09-13T11:37:41.762514Z",
#     "end_time": "2024-09-13T11:37:43.806178Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "temperature": 0.1,
#         "model_name": "gpt-4",
#         "provider_name": "api.openai.com",
#         "inference_endpoint": "https://api.openai.com/v1",
#         "completion_tokens": 15,
#         "prompt_tokens": 142,
#         "total_tokens": 157
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
#     "name": "llamaindex.openai",
#     "context": {
#         "trace_id": "0x79de813d316bd63f40767fec67560d0f",
#         "span_id": "0x99c6a02cfdfddb24",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa9e16efccceb9368",
#     "start_time": "2024-09-13T11:37:41.762330Z",
#     "end_time": "2024-09-13T11:37:43.806247Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "temperature": 0.1,
#         "model_name": "gpt-4",
#         "provider_name": "api.openai.com",
#         "inference_endpoint": "https://api.openai.com/v1",
#         "completion_tokens": 15,
#         "prompt_tokens": 142,
#         "total_tokens": 157
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
#     "name": "llamaindex.query",
#     "context": {
#         "trace_id": "0x79de813d316bd63f40767fec67560d0f",
#         "span_id": "0xa9e16efccceb9368",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-13T11:37:40.848077Z",
#     "end_time": "2024-09-13T11:37:43.806928Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "tags": [
#             "text-embedding-3-large",
#             "ChromaVectorStore"
#         ],
#         "type": "vector_store",
#         "provider_name": "ChromaVectorStore",
#         "embedding_model": "text-embedding-3-large",
#         "workflow_name": "llama_index_1",
#         "workflow_type": "workflow.llamaindex"
#     },
#     "events": [
#         {
#             "name": "input",
#             "timestamp": "2024-09-13T11:37:40.848123Z",
#             "attributes": {
#                 "question": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "output",
#             "timestamp": "2024-09-13T11:37:43.806899Z",
#             "attributes": {
#                 "response": "The context does not provide information about what the author did while growing up."
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


