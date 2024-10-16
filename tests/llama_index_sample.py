

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
#         "trace_id": "0x9b384dc0255183ffacee26e2da0d8655",
#         "span_id": "0x8974f60f72d5d0ad",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0f469ddf63b1b514",
#     "start_time": "2024-10-16T13:08:17.828595Z",
#     "end_time": "2024-10-16T13:08:18.411710Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "ChromaVectorStore",
#         "entity.1.type": "vectorstore.ChromaVectorStore",
#         "entity.1.embedding_model_name": "text-embedding-3-large",
#         "entity.2.name": "text-embedding-3-large",
#         "entity.2.type": "model.embedding",
#         "entity.2.model_name": "text-embedding-3-large"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:08:17.829609Z",
#             "attributes": {
#                 "question": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:08:18.411710Z",
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
#         "trace_id": "0x9b384dc0255183ffacee26e2da0d8655",
#         "span_id": "0x9c6d57bb612795db",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0f469ddf63b1b514",
#     "start_time": "2024-10-16T13:08:18.412710Z",
#     "end_time": "2024-10-16T13:08:19.842145Z",
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
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-4"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-10-16T13:08:19.842145Z",
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
#         "trace_id": "0x9b384dc0255183ffacee26e2da0d8655",
#         "span_id": "0x0f469ddf63b1b514",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-10-16T13:08:17.828595Z",
#     "end_time": "2024-10-16T13:08:19.842145Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "workflow_name": "llama_index_1",
#         "workflow_type": "workflow.llamaindex"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:08:17.828595Z",
#             "attributes": {
#                 "question": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:08:17.828595Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:08:19.842145Z",
#             "attributes": {
#                 "response": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:08:19.842145Z",
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