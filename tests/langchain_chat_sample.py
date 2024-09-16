

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
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0x036011bfdfdcb90a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7afa7a66a2adfb4a",
#     "start_time": "2024-06-10T04:38:55.693625Z",
#     "end_time": "2024-06-10T04:38:56.241083Z",
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
#             "timestamp": "2024-09-16T09:48:53.462202Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0x7afa7a66a2adfb4a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x536c28587fc639a8",
#     "start_time": "2024-06-10T04:38:55.692022Z",
#     "end_time": "2024-06-10T04:38:56.241167Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0x536c28587fc639a8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf38e594bba842099",
#     "start_time": "2024-06-10T04:38:55.686227Z",
#     "end_time": "2024-06-10T04:38:56.241965Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0x54fa0fc40129d7c8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xfd62c1c2c9d666ed",
#     "start_time": "2024-06-10T04:38:56.268526Z",
#     "end_time": "2024-06-10T04:38:56.270750Z",
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
# },
# {
#     "name": "langchain.task.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0xcc431732937f7052",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xfd62c1c2c9d666ed",
#     "start_time": "2024-06-10T04:38:56.270832Z",
#     "end_time": "2024-06-10T04:38:56.271675Z",
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
# },
# {
#     "name": "langchain.task.ChatOpenAI",
#     "context": {
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0x55453deb49cda82d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xfd62c1c2c9d666ed",
#     "start_time": "2024-06-10T04:38:56.271747Z",
#     "end_time": "2024-06-10T04:38:57.914210Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "server_url": "http://triton22.eastus.cloudapp.azure.com:8000/v2/models/flan_t5_783m/versions/1/infer",
#         "completion_tokens": 57,
#         "prompt_tokens": 580,
#         "total_tokens": 637
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain.task.StrOutputParser",
#     "context": {
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0x32539134995fccec",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xfd62c1c2c9d666ed",
#     "start_time": "2024-06-10T04:38:57.914369Z",
#     "end_time": "2024-06-10T04:38:57.914929Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0xfd62c1c2c9d666ed",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xfcc716b485539b93",
#     "start_time": "2024-06-10T04:38:56.261349Z",
#     "end_time": "2024-06-10T04:38:57.914961Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0xfcc716b485539b93",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xf38e594bba842099",
#     "start_time": "2024-06-10T04:38:56.253582Z",
#     "end_time": "2024-06-10T04:38:57.915145Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xca3159edb8ac4ba9fd87ba54aa5df4aa",
#         "span_id": "0xf38e594bba842099",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "None",
#     "start_time": "2024-06-10T04:38:55.640160Z",
#     "end_time": "2024-06-10T04:38:57.915229Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "workflow_input": "What is Task Decomposition?",
#         "workflow_name": "langchain_app_1",
#         "workflow_output": "Task decomposition is a technique used to break down complex tasks into smaller and more manageable steps. This process helps agents or models handle intricate tasks by dividing them into simpler subtasks. Various methods, such as Chain of Thought and Tree of Thoughts, can be employed to decompose tasks effectively.",
#         "workflow_type": "workflow.langchain"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain.task.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0xa3ae254e712e3f90",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa9b366f5c4fb2eda",
#     "start_time": "2024-06-10T04:38:57.941590Z",
#     "end_time": "2024-06-10T04:38:57.942342Z",
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
# },
# {
#     "name": "langchain.task.ChatOpenAI",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0x419b04f8a3eb4883",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa9b366f5c4fb2eda",
#     "start_time": "2024-06-10T04:38:57.942406Z",
#     "end_time": "2024-06-10T04:38:59.211431Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "server_url": "http://triton22.eastus.cloudapp.azure.com:8000/v2/models/flan_t5_783m/versions/1/infer",
#         "completion_tokens": 10,
#         "prompt_tokens": 140,
#         "total_tokens": 150
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain.task.StrOutputParser",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0xaaa3a958fb1da0e9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa9b366f5c4fb2eda",
#     "start_time": "2024-06-10T04:38:59.211922Z",
#     "end_time": "2024-06-10T04:38:59.213538Z",
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
# },
# {
#     "name": "langchain.task.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0x3e8142ee7d8d4927",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa9b366f5c4fb2eda",
#     "start_time": "2024-06-10T04:38:59.213754Z",
#     "end_time": "2024-06-10T04:38:59.699996Z",
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
#             "timestamp": "2024-09-16T09:48:56.731819Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0xa9b366f5c4fb2eda",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xefdcdb61e167f73a",
#     "start_time": "2024-06-10T04:38:57.940414Z",
#     "end_time": "2024-06-10T04:38:59.700076Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0xefdcdb61e167f73a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xffdc0a0d41b85218",
#     "start_time": "2024-06-10T04:38:57.934140Z",
#     "end_time": "2024-06-10T04:38:59.700674Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0xa0b015ed781ad960",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3711b72dfa932d3e",
#     "start_time": "2024-06-10T04:38:59.726886Z",
#     "end_time": "2024-06-10T04:38:59.729179Z",
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
# },
# {
#     "name": "langchain.task.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0x0768296ba09b7230",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3711b72dfa932d3e",
#     "start_time": "2024-06-10T04:38:59.729256Z",
#     "end_time": "2024-06-10T04:38:59.730086Z",
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
# },
# {
#     "name": "langchain.task.ChatOpenAI",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0xa32f64207539d7a8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3711b72dfa932d3e",
#     "start_time": "2024-06-10T04:38:59.730152Z",
#     "end_time": "2024-06-10T04:39:01.261308Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "server_url": "http://triton22.eastus.cloudapp.azure.com:8000/v2/models/flan_t5_783m/versions/1/infer",
#         "completion_tokens": 63,
#         "prompt_tokens": 619,
#         "total_tokens": 682
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain.task.StrOutputParser",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0xb664f045c3716fa3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3711b72dfa932d3e",
#     "start_time": "2024-06-10T04:39:01.261566Z",
#     "end_time": "2024-06-10T04:39:01.262450Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0x3711b72dfa932d3e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0a6e7fac9826a16c",
#     "start_time": "2024-06-10T04:38:59.719843Z",
#     "end_time": "2024-06-10T04:39:01.262503Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0x0a6e7fac9826a16c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xffdc0a0d41b85218",
#     "start_time": "2024-06-10T04:38:59.712013Z",
#     "end_time": "2024-06-10T04:39:01.262831Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"
#     },
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xfcb89e0c5f4aba8a1377664f6dee7661",
#         "span_id": "0xffdc0a0d41b85218",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "None",
#     "start_time": "2024-06-10T04:38:57.915422Z",
#     "end_time": "2024-06-10T04:39:01.262926Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "workflow_input": "What are common ways of doing it?",
#         "workflow_name": "langchain_app_1",
#         "workflow_output": "Task decomposition can be achieved through methods such as using Language Model (LLM) prompting with specific instructions like \"Steps for XYZ\" or \"What are the subgoals for achieving XYZ?\", providing task-specific instructions, or incorporating human inputs. These approaches help in breaking down tasks into smaller components for easier handling and execution.",
#         "workflow_type": "workflow.langchain"
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