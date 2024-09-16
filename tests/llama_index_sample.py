

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
#     "trace_id": "0xbd54e5d0edcd96634fa8a02c25c27519",
#     "start_time": "2024-04-15T23:27:54.806477Z",
#     "end_time": "2024-04-15T23:27:57.182261Z",
#     "duration_ms": "2376",
#     "spans": [
#         {
#             "span_name": "llamaindex.retrieve",
#             "start_time": "2024-04-15T23:27:54.806773Z",
#             "end_time": "2024-04-15T23:27:55.732604Z",
#             "duration_ms": "926",
#             "span_id": "0x030cf03872d4a092",
#             "trace_id": "0xbd54e5d0edcd96634fa8a02c25c27519",
#             "parent_id": "0xb4b14a8f14e7e770",
#             "attributes": {
#             },
#             "events": []
#         },
#         {
#             "span_name": "llamaindex.openai",
#             "start_time": "2024-04-15T23:27:55.740299Z",
#             "end_time": "2024-04-15T23:27:57.181992Z",
#             "duration_ms": "1442",
#             "span_id": "0x225fbfb58481e58c",
#             "trace_id": "0xbd54e5d0edcd96634fa8a02c25c27519",
#             "parent_id": "0xb4b14a8f14e7e770",
#             "attributes": {
#                 "model_name": "gpt-3.5-turbo-0125",
#                 "provider_name": "openai.com",
#             },
#             "events": []
#         },
#         {
#             "span_name": "llamaindex.query",
#             "start_time": "2024-04-15T23:27:54.806477Z",
#             "end_time": "2024-04-15T23:27:57.182261Z",
#             "duration_ms": "2376",
#             "span_id": "0xb4b14a8f14e7e770",
#             "trace_id": "0xbd54e5d0edcd96634fa8a02c25c27519",
#             "parent_id": "None",
#             "attributes": {
#                    "tags": [
#                       "text-embedding-3-large",
#                        "ChromaVectorStore"
#                    ],
#                   "type": "vector_store",
#                   "provider_name": "ChromaVectorStore",
#                   "embedding_model": "text-embedding-3-large",
#                   "workflow_name": "llama_index_1",
#                   "workflow_type": "workflow.llamaindex"
#                   },
#           "events": [
#               {
#                   "name": "input",
#                   "timestamp": "2024-09-16T10:05:44.687175Z",
#                   "attributes": {
#                       "question": "What did the author do growing up?"
#                   }
#               },
#               {
#                   "name": "output",
#                   "timestamp": "2024-09-16T10:05:47.345643Z",
#                   "attributes": {
#                       "response": "The context does not provide information about what the author did while growing up."
#                       }
#                   }
#               ],
#           "links": [],
#           "resource": {
#               "attributes": {
#                   "service.name": "llama_index_1"
#               },
#               "schema_url": ""
#              }
#         }
#     ]
# }


