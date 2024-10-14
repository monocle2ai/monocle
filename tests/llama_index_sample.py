

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
#     "name": "llamaindex.retrieve",
#     "context": {
#         "trace_id": "0x939aa2e13c3ce5b37c74b63dc7cfb163",
#         "span_id": "0x4249f1d3557d62db",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x15eb14200cf48548",
#     "start_time": "2024-09-18T09:11:10.380222Z",
#     "end_time": "2024-09-18T09:11:11.159369Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "tags": [
#             "OpenAIEmbedding",
#             "ChromaVectorStore"
#         ],
#         "type": "vector_store",
#         "provider_name": "ChromaVectorStore",
#         "embedding_model": "text-embedding-3-large"
#     },
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-10-03T12:17:37.780668Z",
#             "attributes": {
#                 "question": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "context_output",
#             "timestamp": "2024-10-03T12:17:38.509564Z",
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
# },
# {
#     "name": "llamaindex.openai",
#     "context": {
#         "trace_id": "0x939aa2e13c3ce5b37c74b63dc7cfb163",
#         "span_id": "0x32754f3f46059db0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x15eb14200cf48548",
#     "start_time": "2024-09-18T09:11:11.161538Z",
#     "end_time": "2024-09-18T09:11:12.893143Z",
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
# },
# {
#     "name": "llamaindex.query",
#     "context": {
#         "trace_id": "0x939aa2e13c3ce5b37c74b63dc7cfb163",
#         "span_id": "0x15eb14200cf48548",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-18T09:11:10.379910Z",
#     "end_time": "2024-09-18T09:11:12.894191Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "tags": [
#             "text-embedding-3-large",
#             "ChromaVectorStore"
#         ],
#         "workflow_name": "llama_index_1",
#         "workflow_type": "workflow.llamaindex"
#     },
#     "events": [
#         {
#             "name": "input",
#             "timestamp": "2024-09-18T09:11:10.379937Z",
#             "attributes": {
#                 "question": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "output",
#             "timestamp": "2024-09-18T09:11:12.894146Z",
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


