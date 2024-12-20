import os
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import OpenSearchVectorSearch  # Change this import
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from monocle.tests.common.langhchain_patch import create_history_aware_retriever
from monocle_apptrace.instrumentation.common.instrumentor import set_context_properties
from langchain_openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0xd376212dbdfb2527",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x2b7a8fe28ba5cc2b",
#     "start_time": "2024-11-26T09:47:21.318322Z",
#     "end_time": "2024-11-26T09:47:23.806760Z",
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
#             "timestamp": "2024-11-26T09:47:21.318322Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-26T09:47:23.806760Z",
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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0x2b7a8fe28ba5cc2b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3266dfe13f8684f2",
#     "start_time": "2024-11-26T09:47:21.317322Z",
#     "end_time": "2024-11-26T09:47:23.806760Z",
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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0x3266dfe13f8684f2",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe2bbefbc429da27d",
#     "start_time": "2024-11-26T09:47:21.316323Z",
#     "end_time": "2024-11-26T09:47:23.818090Z",
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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0x839cf68b99143f5e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x92f1ddda941a02bd",
#     "start_time": "2024-11-26T09:47:23.821100Z",
#     "end_time": "2024-11-26T09:47:23.821100Z",
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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0x3608fbb48969a98d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x92f1ddda941a02bd",
#     "start_time": "2024-11-26T09:47:23.821100Z",
#     "end_time": "2024-11-26T09:47:23.822100Z",
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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0xe51f76fc0c5f71a4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x92f1ddda941a02bd",
#     "start_time": "2024-11-26T09:47:23.822100Z",
#     "end_time": "2024-11-26T09:47:24.858404Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0xb6320a370f0bab0e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x92f1ddda941a02bd",
#     "start_time": "2024-11-26T09:47:24.858404Z",
#     "end_time": "2024-11-26T09:47:24.858404Z",
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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0x92f1ddda941a02bd",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7b580ea444f4836c",
#     "start_time": "2024-11-26T09:47:23.820105Z",
#     "end_time": "2024-11-26T09:47:24.858404Z",
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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0x7b580ea444f4836c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe2bbefbc429da27d",
#     "start_time": "2024-11-26T09:47:23.819099Z",
#     "end_time": "2024-11-26T09:47:24.858404Z",
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
#         "trace_id": "0xea147077112e46a37f532ac50d0202c5",
#         "span_id": "0xe2bbefbc429da27d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-26T09:47:21.291928Z",
#     "end_time": "2024-11-26T09:47:24.858404Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "monocle_apptrace.version": "0.2.0",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_opensearch",
#         "entity.1.type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-26T09:47:21.292928Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": []
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-26T09:47:24.858404Z",
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