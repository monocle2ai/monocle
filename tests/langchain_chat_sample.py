

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
#     "name": "haystack.tracing.auto_enable",
#     "context": {
#         "trace_id": "0xa6129cbc4adb0e601ad9f0569a591613",
#         "span_id": "0x4fb64308d7261b4a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-13T11:53:46.324551Z",
#     "end_time": "2024-09-13T11:53:46.325633Z",
#     "status": {
#         "status_code": "ERROR",
#         "description": "ImportError: cannot import name 'Span' from partially initialized module 'haystack.tracing' (most likely due to a circular import) (/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/__init__.py)"
#     },
#     "attributes": {},
#     "events": [
#         {
#             "name": "exception",
#             "timestamp": "2024-09-13T11:53:46.325617Z",
#             "attributes": {
#                 "exception.type": "ImportError",
#                 "exception.message": "cannot import name 'Span' from partially initialized module 'haystack.tracing' (most likely due to a circular import) (/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/__init__.py)",
#                 "exception.stacktrace": "Traceback (most recent call last):\n  File \"/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/opentelemetry/trace/__init__.py\", line 590, in use_span\n    yield span\n  File \"/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/opentelemetry/sdk/trace/__init__.py\", line 1108, in start_as_current_span\n    yield span\n  File \"/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/tracer.py\", line 207, in _auto_configured_opentelemetry_tracer\n    from haystack.tracing.opentelemetry import OpenTelemetryTracer\n  File \"/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/opentelemetry.py\", line 9, in <module>\n    from haystack.tracing import Span, Tracer\nImportError: cannot import name 'Span' from partially initialized module 'haystack.tracing' (most likely due to a circular import) (/home/beehyv/Documents/monocle/venv/lib/python3.10/site-packages/haystack/tracing/__init__.py)\n",
#                 "exception.escaped": "False"
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
#     "name": "haystack.tracing.auto_enable",
#     "context": {
#         "trace_id": "0xb98697469cb0b72734a2db8e3e7c8d90",
#         "span_id": "0x65f5cb09e1240da0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-13T11:53:47.765068Z",
#     "end_time": "2024-09-13T11:53:47.765132Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0xb6a2ad7ab055eaa8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x5a377a89c97b361f",
#     "start_time": "2024-09-13T11:53:54.945261Z",
#     "end_time": "2024-09-13T11:53:55.622886Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "tags": [
#             "Chroma",
#             "OpenAIEmbeddings"
#         ],
#         "type": "vector_store",
#         "provider_name": "OpenAIEmbeddings",
#         "embedding_model": "Chroma"
#     },
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-09-13T11:53:54.945300Z",
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0x5a377a89c97b361f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd5e3bd8abd3bd3b4",
#     "start_time": "2024-09-13T11:53:54.943933Z",
#     "end_time": "2024-09-13T11:53:55.622959Z",
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0xd5e3bd8abd3bd3b4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7577b1ddc83509cf",
#     "start_time": "2024-09-13T11:53:54.938056Z",
#     "end_time": "2024-09-13T11:53:55.623621Z",
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0x31a2623e96243933",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xce9266e22f7c05f3",
#     "start_time": "2024-09-13T11:53:55.652987Z",
#     "end_time": "2024-09-13T11:53:55.654604Z",
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0x4aa5fd5c16119390",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xce9266e22f7c05f3",
#     "start_time": "2024-09-13T11:53:55.654711Z",
#     "end_time": "2024-09-13T11:53:55.655612Z",
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0xb1dade5154633d8a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xce9266e22f7c05f3",
#     "start_time": "2024-09-13T11:53:55.655688Z",
#     "end_time": "2024-09-13T11:53:57.997859Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "temperature": 0.7,
#         "model_name": "gpt-3.5-turbo-0125",
#         "provider_name": "api.openai.com",
#         "completion_tokens": 62,
#         "prompt_tokens": 580,
#         "total_tokens": 642
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0x61ed8cef55304a27",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xce9266e22f7c05f3",
#     "start_time": "2024-09-13T11:53:57.998110Z",
#     "end_time": "2024-09-13T11:53:57.998636Z",
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0xce9266e22f7c05f3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7ad3c9118ba28f72",
#     "start_time": "2024-09-13T11:53:55.646757Z",
#     "end_time": "2024-09-13T11:53:57.998664Z",
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0x7ad3c9118ba28f72",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7577b1ddc83509cf",
#     "start_time": "2024-09-13T11:53:55.640859Z",
#     "end_time": "2024-09-13T11:53:57.998842Z",
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
#         "trace_id": "0xf490e6be306375354c564c94075ed8df",
#         "span_id": "0x7577b1ddc83509cf",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-13T11:53:54.892772Z",
#     "end_time": "2024-09-13T11:53:57.998937Z",
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
#             "name": "input",
#             "timestamp": "2024-09-13T11:53:54.892847Z",
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0xd6c40115270fbe7a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x342dd0dfd45770e4",
#     "start_time": "2024-09-13T11:53:58.028033Z",
#     "end_time": "2024-09-13T11:53:58.028906Z",
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0xf772bf22bc9316c6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x342dd0dfd45770e4",
#     "start_time": "2024-09-13T11:53:58.028981Z",
#     "end_time": "2024-09-13T11:53:59.002471Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "temperature": 0.7,
#         "model_name": "gpt-3.5-turbo-0125",
#         "provider_name": "api.openai.com",
#         "completion_tokens": 9,
#         "prompt_tokens": 145,
#         "total_tokens": 154
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0xded31022b03dc5c6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x342dd0dfd45770e4",
#     "start_time": "2024-09-13T11:53:59.002761Z",
#     "end_time": "2024-09-13T11:53:59.003855Z",
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0x9110963dc0694b6d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x342dd0dfd45770e4",
#     "start_time": "2024-09-13T11:53:59.004127Z",
#     "end_time": "2024-09-13T11:53:59.578593Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "tags": [
#             "Chroma",
#             "OpenAIEmbeddings"
#         ],
#         "type": "vector_store",
#         "provider_name": "OpenAIEmbeddings",
#         "embedding_model": "Chroma"
#     },
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-09-13T11:53:59.004235Z",
#             "attributes": {
#                 "question": "What are some typical methods for task decomposition?"
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0x342dd0dfd45770e4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x376010a99723e8f6",
#     "start_time": "2024-09-13T11:53:58.026890Z",
#     "end_time": "2024-09-13T11:53:59.578649Z",
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0x376010a99723e8f6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6271cda8f5cf74cc",
#     "start_time": "2024-09-13T11:53:58.020961Z",
#     "end_time": "2024-09-13T11:53:59.579314Z",
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0x3abe3e6f01faf7d3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe78159af24754221",
#     "start_time": "2024-09-13T11:53:59.603308Z",
#     "end_time": "2024-09-13T11:53:59.604992Z",
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0x932b5b9249809e07",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe78159af24754221",
#     "start_time": "2024-09-13T11:53:59.605136Z",
#     "end_time": "2024-09-13T11:53:59.606022Z",
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
# Common ways of task decomposition include using techniques like Chain of Thought (CoT) and Tree of Thoughts, which break down tasks into smaller steps for easier execution. Task decomposition can also be done through simple prompting using language models, task-specific instructions tailored to the specific task, or by incorporating human inputs to guide the decomposition process.
# {
#     "name": "langchain.task.ChatOpenAI",
#     "context": {
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0x95bb04f2be7f0ef0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe78159af24754221",
#     "start_time": "2024-09-13T11:53:59.606117Z",
#     "end_time": "2024-09-13T11:54:01.052883Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "temperature": 0.7,
#         "model_name": "gpt-3.5-turbo-0125",
#         "provider_name": "api.openai.com",
#         "completion_tokens": 65,
#         "prompt_tokens": 658,
#         "total_tokens": 723
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0xf08d575527e91af2",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe78159af24754221",
#     "start_time": "2024-09-13T11:54:01.053197Z",
#     "end_time": "2024-09-13T11:54:01.054195Z",
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0xe78159af24754221",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x29c21e517e9a247a",
#     "start_time": "2024-09-13T11:53:59.597366Z",
#     "end_time": "2024-09-13T11:54:01.054250Z",
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0x29c21e517e9a247a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6271cda8f5cf74cc",
#     "start_time": "2024-09-13T11:53:59.592121Z",
#     "end_time": "2024-09-13T11:54:01.054516Z",
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
#         "trace_id": "0x460ddbe0096e740e5fce324188e8b783",
#         "span_id": "0x6271cda8f5cf74cc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-13T11:53:57.999167Z",
#     "end_time": "2024-09-13T11:54:01.054623Z",
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
#             "name": "input",
#             "timestamp": "2024-09-13T11:53:57.999329Z",
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