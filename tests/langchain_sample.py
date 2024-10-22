

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

setup_monocle_telemetry(
            workflow_name="langchain_app_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[])

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result =  rag_chain.invoke("What is Task Decomposition?")

print(result)

# {
#     "name": "langchain.task.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0xb203db961f9560508f72723e3253c939",
#         "span_id": "0xc0149ffab36124a4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x495270d24f4e77ea",
#     "start_time": "2024-10-16T08:19:03.415255Z",
#     "end_time": "2024-10-16T08:19:03.953946Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "tags": [
#             "Chroma",
#             "OpenAIEmbeddings"
#         ],
#         "type": "vector_store",
#         "provider_name": "Chroma",
#         "embedding_model": "text-embedding-ada-002"
#     },
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-10-16T08:19:03.415374Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "context_output",
#             "timestamp": "2024-10-16T08:19:03.953800Z",
#             "attributes": {
#                 "response": "Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated ta..."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xb203db961f9560508f72723e3253c939",
#         "span_id": "0x495270d24f4e77ea",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0151992fa0b78fb0",
#     "start_time": "2024-10-16T08:19:03.413045Z",
#     "end_time": "2024-10-16T08:19:03.956629Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-10-16T08:19:03.413078Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "context_output",
#             "timestamp": "2024-10-16T08:19:03.956593Z",
#             "attributes": {
#                 "response": ""
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xb203db961f9560508f72723e3253c939",
#         "span_id": "0x0151992fa0b78fb0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x358413b16f81182c",
#     "start_time": "2024-10-16T08:19:03.408637Z",
#     "end_time": "2024-10-16T08:19:03.957381Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-10-16T08:19:03.408679Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "context_output",
#             "timestamp": "2024-10-16T08:19:03.957349Z",
#             "attributes": {
#                 "response": ""
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.task.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0xb203db961f9560508f72723e3253c939",
#         "span_id": "0xe3bac282787e737a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x358413b16f81182c",
#     "start_time": "2024-10-16T08:19:03.957607Z",
#     "end_time": "2024-10-16T08:19:03.960372Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-10-16T08:19:03.957681Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "context_output",
#             "timestamp": "2024-10-16T08:19:03.960347Z",
#             "attributes": {
#                 "response": ""
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.task.ChatOpenAI",
#     "context": {
#         "trace_id": "0xb203db961f9560508f72723e3253c939",
#         "span_id": "0xcb19e499d447a2b9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x358413b16f81182c",
#     "start_time": "2024-10-16T08:19:03.960601Z",
#     "end_time": "2024-10-16T08:19:05.344624Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "temperature": 0.7,
#         "model_name": "gpt-3.5-turbo-0125",
#         "provider_name": "api.openai.com",
#         "completion_tokens": 59,
#         "prompt_tokens": 584,
#         "total_tokens": 643
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.task.StrOutputParser",
#     "context": {
#         "trace_id": "0xb203db961f9560508f72723e3253c939",
#         "span_id": "0xcf3973c932167969",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x358413b16f81182c",
#     "start_time": "2024-10-16T08:19:05.344774Z",
#     "end_time": "2024-10-16T08:19:05.345719Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-10-16T08:19:05.344818Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "context_output",
#             "timestamp": "2024-10-16T08:19:05.345703Z",
#             "attributes": {
#                 "response": ""
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xb203db961f9560508f72723e3253c939",
#         "span_id": "0x358413b16f81182c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-10-16T08:19:03.295884Z",
#     "end_time": "2024-10-16T08:19:05.345881Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "workflow_name": "langchain_app_1",
#         "workflow_type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "input",
#             "timestamp": "2024-10-16T08:19:03.296479Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "context_input",
#             "timestamp": "2024-10-16T08:19:03.296905Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "context_output",
#             "timestamp": "2024-10-16T08:19:05.345766Z",
#             "attributes": {
#                 "response": ""
#             }
#         },
#         {
#             "name": "output",
#             "timestamp": "2024-10-16T08:19:05.345845Z",
#             "attributes": {
#                 "response": "Task Decomposition is a technique where complex tasks are broken down into smaller and simpler steps to enhance model performance. This process helps in transforming big tasks into multiple manageable tasks by thinking step by step. Task decomposition can be achieved through prompting techniques using LLM, task-specific instructions, or human inputs."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }