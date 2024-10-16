
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
import logging
logging.basicConfig(level=logging.INFO)

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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0x7040ef70bc35e241",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x5287ab2cc57c0f73",
#     "start_time": "2024-10-16T14:40:24.950580Z",
#     "end_time": "2024-10-16T14:40:26.356567Z",
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
#             "timestamp": "2024-10-16T14:40:24.951586Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T14:40:26.356567Z",
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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0x5287ab2cc57c0f73",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x5a0ba6c1f6749bc5",
#     "start_time": "2024-10-16T14:40:24.949575Z",
#     "end_time": "2024-10-16T14:40:26.356567Z",
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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0x5a0ba6c1f6749bc5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x19812aa8965570f5",
#     "start_time": "2024-10-16T14:40:24.943918Z",
#     "end_time": "2024-10-16T14:40:26.356567Z",
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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0x9848fc0bd934fd8e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x318e0e2d49327f12",
#     "start_time": "2024-10-16T14:40:26.374844Z",
#     "end_time": "2024-10-16T14:40:26.376840Z",
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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0xc1770a24023d8770",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x318e0e2d49327f12",
#     "start_time": "2024-10-16T14:40:26.376840Z",
#     "end_time": "2024-10-16T14:40:26.377844Z",
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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0x8abd8a67b029c603",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x318e0e2d49327f12",
#     "start_time": "2024-10-16T14:40:26.377844Z",
#     "end_time": "2024-10-16T14:40:28.964247Z",
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
#             "timestamp": "2024-10-16T14:40:28.964247Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 73,
#                 "prompt_tokens": 580,
#                 "total_tokens": 653
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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0x59010abbd3ad5f07",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x318e0e2d49327f12",
#     "start_time": "2024-10-16T14:40:28.964247Z",
#     "end_time": "2024-10-16T14:40:28.964247Z",
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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0x318e0e2d49327f12",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x539b1b9b49a9caa6",
#     "start_time": "2024-10-16T14:40:26.369842Z",
#     "end_time": "2024-10-16T14:40:28.964247Z",
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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0x539b1b9b49a9caa6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x19812aa8965570f5",
#     "start_time": "2024-10-16T14:40:26.364330Z",
#     "end_time": "2024-10-16T14:40:28.964247Z",
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
#         "trace_id": "0x8d06a36f77faccfb46b102dc4201bb62",
#         "span_id": "0x19812aa8965570f5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-10-16T14:40:24.907692Z",
#     "end_time": "2024-10-16T14:40:28.965245Z",
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
#             "timestamp": "2024-10-16T14:40:24.907692Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": []
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T14:40:28.965245Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": [],
#                 "answer": "Task decomposition involves breaking down complex tasks into smaller and simpler steps to make them more manageable and easier to solve. Techniques like Chain of Thought and Tree of Thoughts help agents or models decompose hard tasks into multiple subgoals or thoughts, enhancing performance on complex tasks. Task decomposition can be achieved through simple prompting, task-specific instructions, or human inputs to guide the process."
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x453cbe6e04b6a478",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4cba04f212eeaf73",
#     "start_time": "2024-10-16T14:40:28.987068Z",
#     "end_time": "2024-10-16T14:40:28.987068Z",
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x6e436f96bd1c8432",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4cba04f212eeaf73",
#     "start_time": "2024-10-16T14:40:28.987068Z",
#     "end_time": "2024-10-16T14:40:29.954097Z",
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
#             "timestamp": "2024-10-16T14:40:29.954097Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 8,
#                 "prompt_tokens": 156,
#                 "total_tokens": 164
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x7a7a7096ff334224",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4cba04f212eeaf73",
#     "start_time": "2024-10-16T14:40:29.954097Z",
#     "end_time": "2024-10-16T14:40:29.954097Z",
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x0f0628652c73e7bc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4cba04f212eeaf73",
#     "start_time": "2024-10-16T14:40:29.954097Z",
#     "end_time": "2024-10-16T14:40:30.925542Z",
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
#             "timestamp": "2024-10-16T14:40:29.954097Z",
#             "attributes": {
#                 "question": "What are common methods of task decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T14:40:30.925542Z",
#             "attributes": {
#                 "response": "Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each..."
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x4cba04f212eeaf73",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xfab9e4a936a1122b",
#     "start_time": "2024-10-16T14:40:28.986068Z",
#     "end_time": "2024-10-16T14:40:30.925542Z",
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0xfab9e4a936a1122b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa28ab3d13555dc66",
#     "start_time": "2024-10-16T14:40:28.979758Z",
#     "end_time": "2024-10-16T14:40:30.925542Z",
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0xd73c2b46acb70621",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9a0956a938881cf6",
#     "start_time": "2024-10-16T14:40:30.943736Z",
#     "end_time": "2024-10-16T14:40:30.945741Z",
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x5c85c33b43509c5d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9a0956a938881cf6",
#     "start_time": "2024-10-16T14:40:30.945741Z",
#     "end_time": "2024-10-16T14:40:30.945741Z",
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x7059849660c3b799",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9a0956a938881cf6",
#     "start_time": "2024-10-16T14:40:30.945741Z",
#     "end_time": "2024-10-16T14:40:32.500916Z",
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
#             "timestamp": "2024-10-16T14:40:32.500916Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 95,
#                 "prompt_tokens": 669,
#                 "total_tokens": 764
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x0dfefb758e7b8e43",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9a0956a938881cf6",
#     "start_time": "2024-10-16T14:40:32.500916Z",
#     "end_time": "2024-10-16T14:40:32.501933Z",
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x9a0956a938881cf6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x17f373d254f1b437",
#     "start_time": "2024-10-16T14:40:30.939225Z",
#     "end_time": "2024-10-16T14:40:32.501933Z",
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0x17f373d254f1b437",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa28ab3d13555dc66",
#     "start_time": "2024-10-16T14:40:30.933133Z",
#     "end_time": "2024-10-16T14:40:32.502440Z",
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
#         "trace_id": "0xa3303399d6377c134e8523ae5e0a617e",
#         "span_id": "0xa28ab3d13555dc66",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-10-16T14:40:28.965245Z",
#     "end_time": "2024-10-16T14:40:32.502440Z",
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
#             "timestamp": "2024-10-16T14:40:28.965245Z",
#             "attributes": {
#                 "input": "What are common ways of doing it?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T14:40:32.502440Z",
#             "attributes": {
#                 "input": "What are common ways of doing it?",
#                 "answer": "Task decomposition can be accomplished through prompting using Language Model (LLM) with simple instructions like \"Steps for XYZ\" or \"What are the subgoals for achieving XYZ?\", task-specific instructions such as \"Write a story outline\" for specific tasks like writing a novel, or with human inputs to guide the breakdown of complex tasks into smaller steps. These approaches help in breaking down big tasks into more manageable components and provide a structured way for agents or models to tackle complex problems effectively."
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