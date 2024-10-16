from multiprocessing.forkserver import connect_to_new_process

import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from monocle_apptrace.instrumentor import set_context_properties, setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from langhchain_patch import create_history_aware_retriever
from monocle_apptrace.exporters.aws.s3_exporter import S3SpanExporter
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from dotenv import load_dotenv, dotenv_values
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
exporter = S3SpanExporter(
    region_name='us-east-1',
    bucket_name='sachin-dev'
)

setup_monocle_telemetry(
            workflow_name="langchain_app_1",
            span_processors=[BatchSpanProcessor(exporter)],
            wrapper_methods=[])


llm = ChatOpenAI(model="gpt-3.5-turbo-0125",api_key=OPENAI_API_KEY)


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
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY))

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

chat_history = []

set_context_properties({"session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"})

question = "What is Task Decomposition?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
print(ai_msg_1["answer"])
chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

second_question = "What are common ways of doing it?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

print(ai_msg_2["answer"])


# {
#     "name": "langchain.task.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0x6dcf740981b328cd",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf22bb6b24a47d08e",
#     "start_time": "2024-10-16T13:04:59.514226Z",
#     "end_time": "2024-10-16T13:04:59.796746Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "Chroma",
#         "entity.1.type": "vectorstore.Chroma",
#         "entity.1.embedding_model_name": "text-embedding-ada-002",
#         "entity.2.name": "text-embedding-ada-002",
#         "entity.2.type": "model.embedding",
#         "entity.2.model_name": "text-embedding-ada-002"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:04:59.514226Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:04:59.796746Z",
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
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0xf22bb6b24a47d08e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6928e06fd862a88c",
#     "start_time": "2024-10-16T13:04:59.512690Z",
#     "end_time": "2024-10-16T13:04:59.797735Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:04:59.512690Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:04:59.797735Z",
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
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0x6928e06fd862a88c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf49c89a3b7695614",
#     "start_time": "2024-10-16T13:04:59.506152Z",
#     "end_time": "2024-10-16T13:04:59.797735Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:04:59.506152Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:04:59.797735Z",
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
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0x8675ff5d6a848584",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3a5ee2ec45670a01",
#     "start_time": "2024-10-16T13:04:59.814139Z",
#     "end_time": "2024-10-16T13:04:59.816139Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:04:59.814139Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:04:59.816139Z",
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
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0x8bdd4d1e36a685c3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3a5ee2ec45670a01",
#     "start_time": "2024-10-16T13:04:59.816139Z",
#     "end_time": "2024-10-16T13:04:59.817139Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:04:59.816139Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:04:59.817139Z",
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
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0x01e1e2cbc5188165",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3a5ee2ec45670a01",
#     "start_time": "2024-10-16T13:04:59.817139Z",
#     "end_time": "2024-10-16T13:05:01.503201Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-10-16T13:05:01.503201Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 70,
#                 "prompt_tokens": 580,
#                 "total_tokens": 650
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
#     "name": "langchain.task.StrOutputParser",
#     "context": {
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0x5b214f3b234786a2",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3a5ee2ec45670a01",
#     "start_time": "2024-10-16T13:05:01.504201Z",
#     "end_time": "2024-10-16T13:05:01.504201Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:01.504201Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:01.504201Z",
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
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0x3a5ee2ec45670a01",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe5805ed622d8c943",
#     "start_time": "2024-10-16T13:04:59.809103Z",
#     "end_time": "2024-10-16T13:05:01.504201Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:04:59.809103Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:01.504201Z",
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
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0xe5805ed622d8c943",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf49c89a3b7695614",
#     "start_time": "2024-10-16T13:04:59.804001Z",
#     "end_time": "2024-10-16T13:05:01.504201Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:04:59.804001Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:01.504201Z",
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
#         "trace_id": "0xc5d7ac68a3fc32b683e99dcef160422c",
#         "span_id": "0xf49c89a3b7695614",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-10-16T13:04:59.468714Z",
#     "end_time": "2024-10-16T13:05:01.504201Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "workflow_name": "langchain_app_1",
#         "workflow_type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:04:59.468714Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": []
#             }
#         },
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:04:59.468714Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:01.504201Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0x9d525c40f8695ad6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf69e3b9a83bf6bfb",
#     "start_time": "2024-10-16T13:05:01.525708Z",
#     "end_time": "2024-10-16T13:05:01.526707Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:01.525708Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:01.526707Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0x593db8e5ccea9036",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf69e3b9a83bf6bfb",
#     "start_time": "2024-10-16T13:05:01.526707Z",
#     "end_time": "2024-10-16T13:05:02.501735Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-10-16T13:05:02.501735Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 9,
#                 "prompt_tokens": 153,
#                 "total_tokens": 162
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
#     "name": "langchain.task.StrOutputParser",
#     "context": {
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0xa78f5110f8f45dc1",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf69e3b9a83bf6bfb",
#     "start_time": "2024-10-16T13:05:02.501735Z",
#     "end_time": "2024-10-16T13:05:02.501735Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:02.501735Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:02.501735Z",
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
#     "name": "langchain.task.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0xdd83fe45fe1175ac",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf69e3b9a83bf6bfb",
#     "start_time": "2024-10-16T13:05:02.501735Z",
#     "end_time": "2024-10-16T13:05:03.342518Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "Chroma",
#         "entity.1.type": "vectorstore.Chroma",
#         "entity.1.embedding_model_name": "text-embedding-ada-002",
#         "entity.2.name": "text-embedding-ada-002",
#         "entity.2.type": "model.embedding",
#         "entity.2.model_name": "text-embedding-ada-002"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:02.502740Z",
#             "attributes": {
#                 "question": "What are typical methods used for task decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:03.342518Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0xf69e3b9a83bf6bfb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xb89428628256a7c3",
#     "start_time": "2024-10-16T13:05:01.524703Z",
#     "end_time": "2024-10-16T13:05:03.342518Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:01.524703Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:03.342518Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0xb89428628256a7c3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x1821460947bc7307",
#     "start_time": "2024-10-16T13:05:01.519200Z",
#     "end_time": "2024-10-16T13:05:03.342518Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:01.519200Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:03.342518Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0x22058cc71848b370",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9659d5f599e478bd",
#     "start_time": "2024-10-16T13:05:03.361811Z",
#     "end_time": "2024-10-16T13:05:03.363820Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:03.361811Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:03.363820Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0x01910424c6317fe1",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9659d5f599e478bd",
#     "start_time": "2024-10-16T13:05:03.363820Z",
#     "end_time": "2024-10-16T13:05:03.364824Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:03.363820Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:03.364824Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0x74680f0beb17cf6f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9659d5f599e478bd",
#     "start_time": "2024-10-16T13:05:03.364824Z",
#     "end_time": "2024-10-16T13:05:04.699369Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-10-16T13:05:04.699369Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 64,
#                 "prompt_tokens": 614,
#                 "total_tokens": 678
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
#     "name": "langchain.task.StrOutputParser",
#     "context": {
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0x038dc53c9c12bd26",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9659d5f599e478bd",
#     "start_time": "2024-10-16T13:05:04.699369Z",
#     "end_time": "2024-10-16T13:05:04.699369Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:04.699369Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:04.699369Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0x9659d5f599e478bd",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x74ca7c24b2a6ff00",
#     "start_time": "2024-10-16T13:05:03.357043Z",
#     "end_time": "2024-10-16T13:05:04.699369Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:03.357043Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:04.699369Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0x74ca7c24b2a6ff00",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x1821460947bc7307",
#     "start_time": "2024-10-16T13:05:03.350522Z",
#     "end_time": "2024-10-16T13:05:04.699369Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:03.350522Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:04.699369Z",
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
#         "trace_id": "0x9b427909ca38b77658b50203f15ad504",
#         "span_id": "0x1821460947bc7307",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-10-16T13:05:01.504201Z",
#     "end_time": "2024-10-16T13:05:04.699369Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "workflow_name": "langchain_app_1",
#         "workflow_type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:01.505201Z",
#             "attributes": {
#                 "input": "What are common ways of doing it?"
#             }
#         },
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T13:05:01.505201Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T13:05:04.699369Z",
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