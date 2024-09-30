

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
from openai import api_key
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from langhchain_patch import create_history_aware_retriever
import os
from dotenv import load_dotenv, dotenv_values
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
setup_monocle_telemetry(
            workflow_name="langchain_app_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
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
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

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
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0x59eb918f012f9482",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xca1d1e49f1d6a8b4",
#     "start_time": "2024-09-30T11:16:56.454392Z",
#     "end_time": "2024-09-30T11:16:56.898873Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "Retreival",
#         "entity.count": 2,
#         "entity.1.name": "Chroma",
#         "entity.1.type": "vectorstore.chroma",
#         "entity.1.embedding_model_name": "OpenAIEmbeddings",
#         "entity.2.name": "OpenAIEmbeddings",
#         "entity.2.type": "model.embedding",
#         "entity.2.model_name": "OpenAIEmbeddings",
#         "tags": [
#             "Chroma",
#             "OpenAIEmbeddings"
#         ]
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-09-30T11:16:56.454392Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
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
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0xca1d1e49f1d6a8b4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xcab5b4cde2637a21",
#     "start_time": "2024-09-30T11:16:56.446398Z",
#     "end_time": "2024-09-30T11:16:56.898873Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0xcab5b4cde2637a21",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65c1f51a19847f14",
#     "start_time": "2024-09-30T11:16:56.427210Z",
#     "end_time": "2024-09-30T11:16:56.899890Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0x41745f0cfc13fc37",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7b4343db4e467f34",
#     "start_time": "2024-09-30T11:16:57.004369Z",
#     "end_time": "2024-09-30T11:16:57.012866Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.task.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0x9145bd680c406c6f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7b4343db4e467f34",
#     "start_time": "2024-09-30T11:16:57.012866Z",
#     "end_time": "2024-09-30T11:16:57.015862Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.task.ChatOpenAI",
#     "context": {
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0xd1630e8717810cdd",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7b4343db4e467f34",
#     "start_time": "2024-09-30T11:16:57.016862Z",
#     "end_time": "2024-09-30T11:16:59.318509Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span_type": "Inference",
#         "entities_count": 2,
#         "entity.1.name": "AzureOpenAI",
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "meta_data",
#             "timestamp": "2024-09-30T11:16:59.318509Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 71,
#                 "prompt_tokens": 580,
#                 "total_tokens": 651
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
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0xb2cfccc5224acca2",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7b4343db4e467f34",
#     "start_time": "2024-09-30T11:16:59.319509Z",
#     "end_time": "2024-09-30T11:16:59.320508Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0x7b4343db4e467f34",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x5f81205f6efd49e7",
#     "start_time": "2024-09-30T11:16:56.975277Z",
#     "end_time": "2024-09-30T11:16:59.320508Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0x5f81205f6efd49e7",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65c1f51a19847f14",
#     "start_time": "2024-09-30T11:16:56.938800Z",
#     "end_time": "2024-09-30T11:16:59.321489Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x3a0beb11112d63dc04bbfd9de2e4b2da",
#         "span_id": "0x65c1f51a19847f14",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-30T11:16:56.279733Z",
#     "end_time": "2024-09-30T11:16:59.321489Z",
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
#             "timestamp": "2024-09-30T11:16:56.279733Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": []
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
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x02ac9ed5b80ef187",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc8c9e18577d44e4c",
#     "start_time": "2024-09-30T11:16:59.409943Z",
#     "end_time": "2024-09-30T11:16:59.412943Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.task.ChatOpenAI",
#     "context": {
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x537b88be13b4b30c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc8c9e18577d44e4c",
#     "start_time": "2024-09-30T11:16:59.412943Z",
#     "end_time": "2024-09-30T11:17:00.164070Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span_type": "Inference",
#         "entities_count": 2,
#         "entity.1.name": "AzureOpenAI",
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "meta_data",
#             "timestamp": "2024-09-30T11:17:00.164070Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 8,
#                 "prompt_tokens": 154,
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
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x646c393ede25823d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc8c9e18577d44e4c",
#     "start_time": "2024-09-30T11:17:00.164070Z",
#     "end_time": "2024-09-30T11:17:00.166072Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.task.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x4025622569b73b45",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc8c9e18577d44e4c",
#     "start_time": "2024-09-30T11:17:00.166072Z",
#     "end_time": "2024-09-30T11:17:00.829927Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "Retreival",
#         "entity.count": 2,
#         "entity.1.name": "Chroma",
#         "entity.1.type": "vectorstore.chroma",
#         "entity.1.embedding_model_name": "OpenAIEmbeddings",
#         "entity.2.name": "OpenAIEmbeddings",
#         "entity.2.type": "model.embedding",
#         "entity.2.model_name": "OpenAIEmbeddings",
#         "tags": [
#             "Chroma",
#             "OpenAIEmbeddings"
#         ]
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-09-30T11:17:00.166072Z",
#             "attributes": {
#                 "question": "What are typical methods for task decomposition?"
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
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0xc8c9e18577d44e4c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x43ae6ee7eb5238f9",
#     "start_time": "2024-09-30T11:16:59.407943Z",
#     "end_time": "2024-09-30T11:17:00.829927Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x43ae6ee7eb5238f9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x606136c614f1cc8b",
#     "start_time": "2024-09-30T11:16:59.387103Z",
#     "end_time": "2024-09-30T11:17:00.831927Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x1bd70b64409cbed3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x200b7e52eabb19c7",
#     "start_time": "2024-09-30T11:17:00.924815Z",
#     "end_time": "2024-09-30T11:17:00.935596Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.task.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0xbd3420468a072cea",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x200b7e52eabb19c7",
#     "start_time": "2024-09-30T11:17:00.935596Z",
#     "end_time": "2024-09-30T11:17:00.938956Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.task.ChatOpenAI",
#     "context": {
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x22616f6870c2a140",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x200b7e52eabb19c7",
#     "start_time": "2024-09-30T11:17:00.938956Z",
#     "end_time": "2024-09-30T11:17:02.939039Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span_type": "Inference",
#         "entities_count": 2,
#         "entity.1.name": "AzureOpenAI",
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "meta_data",
#             "timestamp": "2024-09-30T11:17:02.939039Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 82,
#                 "prompt_tokens": 667,
#                 "total_tokens": 749
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
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0xc422c887cde2dfe4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x200b7e52eabb19c7",
#     "start_time": "2024-09-30T11:17:02.939039Z",
#     "end_time": "2024-09-30T11:17:02.940039Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x200b7e52eabb19c7",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x793c0fe9dbe6b317",
#     "start_time": "2024-09-30T11:17:00.899053Z",
#     "end_time": "2024-09-30T11:17:02.941040Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x793c0fe9dbe6b317",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x606136c614f1cc8b",
#     "start_time": "2024-09-30T11:17:00.867414Z",
#     "end_time": "2024-09-30T11:17:02.941040Z",
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
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x68503375dd8c44110426995d17474817",
#         "span_id": "0x606136c614f1cc8b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-30T11:16:59.321489Z",
#     "end_time": "2024-09-30T11:17:02.941040Z",
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
#             "timestamp": "2024-09-30T11:16:59.321489Z",
#             "attributes": {
#                 "input": "What are common ways of doing it?"
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