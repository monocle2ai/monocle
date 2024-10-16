

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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0x2b791bda34bde4fb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x65b3e191154a7aac",
#     "start_time": "2024-10-16T09:52:29.394788Z",
#     "end_time": "2024-10-16T09:52:29.856904Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "retrieval",
#         "span.count": 2,
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
#             "timestamp": "2024-10-16T09:52:29.395788Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:29.856904Z",
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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0x65b3e191154a7aac",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x028750bf9dd63df9",
#     "start_time": "2024-10-16T09:52:29.393785Z",
#     "end_time": "2024-10-16T09:52:29.856904Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:29.393785Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:29.856904Z",
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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0x028750bf9dd63df9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xcf200fa4155404d0",
#     "start_time": "2024-10-16T09:52:29.387789Z",
#     "end_time": "2024-10-16T09:52:29.857409Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:29.387789Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:29.857409Z",
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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0x5f541e14a45742da",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0947546349d0dacd",
#     "start_time": "2024-10-16T09:52:29.875515Z",
#     "end_time": "2024-10-16T09:52:29.877144Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:29.875515Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:29.877144Z",
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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0x7ac47a211284068a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0947546349d0dacd",
#     "start_time": "2024-10-16T09:52:29.877144Z",
#     "end_time": "2024-10-16T09:52:29.878151Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:29.877144Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:29.878151Z",
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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0xad677c3ce14fa738",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0947546349d0dacd",
#     "start_time": "2024-10-16T09:52:29.878151Z",
#     "end_time": "2024-10-16T09:52:31.792647Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "inference",
#         "span.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-10-16T09:52:31.792647Z",
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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0x8ef54d48120b59c3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0947546349d0dacd",
#     "start_time": "2024-10-16T09:52:31.792647Z",
#     "end_time": "2024-10-16T09:52:31.792647Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:31.792647Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:31.792647Z",
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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0x0947546349d0dacd",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x70fb9c42ebbff906",
#     "start_time": "2024-10-16T09:52:29.870485Z",
#     "end_time": "2024-10-16T09:52:31.792647Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:29.870485Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:31.792647Z",
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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0x70fb9c42ebbff906",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xcf200fa4155404d0",
#     "start_time": "2024-10-16T09:52:29.864458Z",
#     "end_time": "2024-10-16T09:52:31.792647Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:29.864458Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:31.792647Z",
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
#         "trace_id": "0x292eea442b46a8111db11587596730b9",
#         "span_id": "0xcf200fa4155404d0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-10-16T09:52:29.353788Z",
#     "end_time": "2024-10-16T09:52:31.792647Z",
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
#             "timestamp": "2024-10-16T09:52:29.353788Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": []
#             }
#         },
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:29.353788Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:31.792647Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0xd5f82f027d485a00",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x49ae78030a0665ef",
#     "start_time": "2024-10-16T09:52:31.815371Z",
#     "end_time": "2024-10-16T09:52:31.816378Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:31.815371Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:31.816378Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0xe470831f25e4b9be",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x49ae78030a0665ef",
#     "start_time": "2024-10-16T09:52:31.816378Z",
#     "end_time": "2024-10-16T09:52:32.455731Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "inference",
#         "span.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-10-16T09:52:32.455731Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 10,
#                 "prompt_tokens": 154,
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0x8227a5cfac917068",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x49ae78030a0665ef",
#     "start_time": "2024-10-16T09:52:32.455731Z",
#     "end_time": "2024-10-16T09:52:32.455731Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:32.455731Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:32.455731Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0x0e49e61326dace1d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x49ae78030a0665ef",
#     "start_time": "2024-10-16T09:52:32.455731Z",
#     "end_time": "2024-10-16T09:52:33.048101Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "retrieval",
#         "span.count": 2,
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
#             "timestamp": "2024-10-16T09:52:32.456731Z",
#             "attributes": {
#                 "question": "What are some typical methods used for task decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:33.048101Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0x49ae78030a0665ef",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0d35c2c23c97b93a",
#     "start_time": "2024-10-16T09:52:31.814367Z",
#     "end_time": "2024-10-16T09:52:33.048101Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:31.814367Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:33.048101Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0x0d35c2c23c97b93a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x34efaf1bc7bb171b",
#     "start_time": "2024-10-16T09:52:31.807851Z",
#     "end_time": "2024-10-16T09:52:33.048101Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:31.807851Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:33.048101Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0xeaae95537c371f3c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xedcc85d7c7369c38",
#     "start_time": "2024-10-16T09:52:33.066993Z",
#     "end_time": "2024-10-16T09:52:33.067997Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:33.066993Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:33.067997Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0x0f749358be3cda29",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xedcc85d7c7369c38",
#     "start_time": "2024-10-16T09:52:33.067997Z",
#     "end_time": "2024-10-16T09:52:33.068997Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:33.067997Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:33.068997Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0x9c68fd7162d94b46",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xedcc85d7c7369c38",
#     "start_time": "2024-10-16T09:52:33.068997Z",
#     "end_time": "2024-10-16T09:52:34.802116Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "inference",
#         "span.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm",
#         "entity.2.model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-10-16T09:52:34.802116Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 84,
#                 "prompt_tokens": 633,
#                 "total_tokens": 717
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0x1233710876b3375d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xedcc85d7c7369c38",
#     "start_time": "2024-10-16T09:52:34.802116Z",
#     "end_time": "2024-10-16T09:52:34.803127Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:34.802116Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:34.803127Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0xedcc85d7c7369c38",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xb0719037837a35e4",
#     "start_time": "2024-10-16T09:52:33.061831Z",
#     "end_time": "2024-10-16T09:52:34.803127Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:33.061831Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:34.803127Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0xb0719037837a35e4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x34efaf1bc7bb171b",
#     "start_time": "2024-10-16T09:52:33.055769Z",
#     "end_time": "2024-10-16T09:52:34.803127Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:33.055769Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:34.803127Z",
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
#         "trace_id": "0x37f4ce7086c0b2421336a2cb539d76bf",
#         "span_id": "0x34efaf1bc7bb171b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-10-16T09:52:31.793647Z",
#     "end_time": "2024-10-16T09:52:34.803127Z",
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
#             "timestamp": "2024-10-16T09:52:31.793647Z",
#             "attributes": {
#                 "input": "What are common ways of doing it?"
#             }
#         },
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-16T09:52:31.793647Z",
#             "attributes": {
#                 "question": ""
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-16T09:52:34.803127Z",
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