import os
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import OpenSearchVectorSearch  # Change this import
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langhchain_patch import create_history_aware_retriever
from monocle_apptrace.instrumentor import set_context_properties
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from monocle_apptrace.instrumentor import setup_monocle_telemetry

# Set up OpenAI API key

# Set up OpenTelemetry tracing
setup_monocle_telemetry(
    workflow_name="langchain_opensearch",
    span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
    wrapper_methods=[]
)

# OpenSearch endpoint and credentials
endpoint = "https://search-sachin-opensearch-cvvd5pdeyrme2l2y26xmcpkm2a.us-east-1.es.amazonaws.com"
http_auth = ("sachin-opensearch", "Sachin@123")
index_name = "gpt-index-demo"

# Initialize OpenSearch client
opensearch_client = OpenSearch(
    hosts=[endpoint],
    http_auth=http_auth,
    connection_class=RequestsHttpConnection
)

# Load documents from a local directory
my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, "data/sample.txt")

loader = TextLoader(data_path)
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Use OpenSearchVectorStore instead of OpenSearchVectorSearch
docsearch = OpenSearchVectorSearch.from_documents(
    docs, embeddings, opensearch_url=endpoint, http_auth=http_auth
)

# Convert to retriever

# Initialize the LLM
llm = OpenAI(temperature=0)
query = "What did the author do growing up?"

retriever = docsearch.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
# This example only specifies a relevant query
chat_history = []

set_context_properties({"session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"})
question = "What is Task Decomposition?"
result = rag_chain.invoke({"input": question, "chat_history": chat_history})

print(result)

# {
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0xccd26ed1f5c6c0a5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x76238767a8366fa5",
#     "start_time": "2024-11-19T12:38:31.982005Z",
#     "end_time": "2024-11-19T12:38:34.084690Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "OpenSearchVectorSearch",
#         "entity.1.type": "vectorstore.OpenSearchVectorSearch",
#         "entity.1.deployment": "https://search-sachin-opensearch-cvvd5pdeyrme2l2y26xmcpkm2a.us-east-1.es.amazonaws.com:443",
#         "entity.2.name": "text-embedding-ada-002",
#         "entity.2.type": "model.embedding.text-embedding-ada-002"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-19T12:38:31.982005Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-19T12:38:34.084690Z",
#             "attributes": {
#                 "response": "this is some sample text"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0x76238767a8366fa5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd44c555547721faf",
#     "start_time": "2024-11-19T12:38:31.980991Z",
#     "end_time": "2024-11-19T12:38:34.084690Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0xd44c555547721faf",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0139030971031e70",
#     "start_time": "2024-11-19T12:38:31.979973Z",
#     "end_time": "2024-11-19T12:38:34.096352Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0xcb51cbd59922d95b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3180028b461cd642",
#     "start_time": "2024-11-19T12:38:34.098406Z",
#     "end_time": "2024-11-19T12:38:34.099421Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0x248cf11f2a9aa278",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3180028b461cd642",
#     "start_time": "2024-11-19T12:38:34.099421Z",
#     "end_time": "2024-11-19T12:38:34.099421Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_openai.llms.base.OpenAI",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0xa77c51379f7da1eb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3180028b461cd642",
#     "start_time": "2024-11-19T12:38:34.100535Z",
#     "end_time": "2024-11-19T12:38:35.811007Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-3.5-turbo-instruct",
#         "entity.2.type": "model.llm.gpt-3.5-turbo-instruct"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.output_parsers.string.StrOutputParser",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0xc082d95472fbbf6f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3180028b461cd642",
#     "start_time": "2024-11-19T12:38:35.811007Z",
#     "end_time": "2024-11-19T12:38:35.811007Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0x3180028b461cd642",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x8eae7672242306a8",
#     "start_time": "2024-11-19T12:38:34.097385Z",
#     "end_time": "2024-11-19T12:38:35.811007Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0x8eae7672242306a8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0139030971031e70",
#     "start_time": "2024-11-19T12:38:34.097385Z",
#     "end_time": "2024-11-19T12:38:35.811007Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x5b0697f5b739c4df0dd7b4bc0c9e6b85",
#         "span_id": "0x0139030971031e70",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-19T12:38:31.970342Z",
#     "end_time": "2024-11-19T12:38:35.812006Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "entity.1.name": "langchain_opensearch",
#         "entity.1.type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-19T12:38:31.970342Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": []
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-19T12:38:35.812006Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": [],
#                 "answer": "\n\nTask Decomposition is a problem-solving strategy that involves breaking down a complex task into smaller, more manageable subtasks. It is often used in artificial intelligence and robotics to solve complex problems. This approach allows for more efficient problem-solving and can lead to better overall performance."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_opensearch"
#         },
#         "schema_url": ""
#     }
# }