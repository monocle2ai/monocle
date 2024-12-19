from os import getenv
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)
import os
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from llama_index.core import VectorStoreIndex, StorageContext

setup_monocle_telemetry(
    workflow_name="llama_index_1",
    span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
    wrapper_methods=[]
)
# http endpoint for your cluster (opensearch required for vector index usage)
endpoint = "https://search-sachin-opensearch-cvvd5pdeyrme2l2y26xmcpkm2a.us-east-1.es.amazonaws.com"
# index to demonstrate the VectorStore impl
idx = "gpt-index-demo"
# load some sample data
my_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(my_path, "data")
documents = SimpleDirectoryReader(model_path).load_data()



# OpensearchVectorClient stores text in this field by default
text_field = "content"
# OpensearchVectorClient stores embeddings in this field by default
embedding_field = "embedding"
# OpensearchVectorClient encapsulates logic for a
# single opensearch index with vector search enabled
client = OpensearchVectorClient(
    endpoint, idx, 1536, embedding_field=embedding_field, text_field=text_field,   http_auth=("sachin-opensearch", "Sachin@123")
)
# initialize vector store
vector_store = OpensearchVectorStore(client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# initialize an index using our sample data and the client we just created
index = VectorStoreIndex.from_documents(
    documents=documents, storage_context=storage_context
)



# run query
query_engine = index.as_query_engine()
res = query_engine.query("What did the author do growing up?")
print(res)

# {
#     "name": "llamaindex.retrieve",
#     "context": {
#         "trace_id": "0x4965082931e44974ed3846094564db6d",
#         "span_id": "0x5fad1c8eacfd5b33",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa54742c93f79aced",
#     "start_time": "2024-11-26T09:49:25.087154Z",
#     "end_time": "2024-11-26T09:49:26.628639Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "OpensearchVectorStore",
#         "entity.1.type": "vectorstore.OpensearchVectorStore",
#         "entity.1.deployment": "https://search-sachin-opensearch-cvvd5pdeyrme2l2y26xmcpkm2a.us-east-1.es.amazonaws.com",
#         "entity.2.name": "text-embedding-ada-002",
#         "entity.2.type": "model.embedding.text-embedding-ada-002"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-26T09:49:25.087154Z",
#             "attributes": {
#                 "question": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-26T09:49:26.628639Z",
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
# The author likely spent time writing and saving text files on their computer.
# {
#     "name": "llamaindex.openai",
#     "context": {
#         "trace_id": "0x4965082931e44974ed3846094564db6d",
#         "span_id": "0x4b8ac2a810c9bd63",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa54742c93f79aced",
#     "start_time": "2024-11-26T09:49:26.630675Z",
#     "end_time": "2024-11-26T09:49:27.588915Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1",
#         "entity.2.name": "gpt-3.5-turbo",
#         "entity.2.type": "model.llm.gpt-3.5-turbo"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-11-26T09:49:27.588915Z",
#             "attributes": {
#                 "temperature": 0.1,
#                 "completion_tokens": 14,
#                 "prompt_tokens": 149,
#                 "total_tokens": 163
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
#         "trace_id": "0x4965082931e44974ed3846094564db6d",
#         "span_id": "0xa54742c93f79aced",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-26T09:49:25.086120Z",
#     "end_time": "2024-11-26T09:49:27.588915Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.2.0",
#         "span.type": "workflow",
#         "entity.1.name": "llama_index_1",
#         "entity.1.type": "workflow.llamaindex"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-26T09:49:25.086120Z",
#             "attributes": {
#                 "question": "What did the author do growing up?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-26T09:49:27.588915Z",
#             "attributes": {
#                 "response": "The author likely spent time writing and saving text files on their computer."
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
