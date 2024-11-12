
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
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0xff1f38688fc91d8a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x1d46f87be140c827",
#     "start_time": "2024-11-12T11:30:11.013606Z",
#     "end_time": "2024-11-12T11:30:11.427990Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "Chroma",
#         "entity.1.type": "vectorstore.Chroma",
#         "entity.2.name": "text-embedding-ada-002",
#         "entity.2.type": "model.embedding.text-embedding-ada-002"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-12T11:30:11.013606Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-12T11:30:11.427990Z",
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
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0x1d46f87be140c827",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x765de37ee36afd67",
#     "start_time": "2024-11-12T11:30:11.013606Z",
#     "end_time": "2024-11-12T11:30:11.427990Z",
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
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0x765de37ee36afd67",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4a827696309fdf1c",
#     "start_time": "2024-11-12T11:30:11.012102Z",
#     "end_time": "2024-11-12T11:30:11.427990Z",
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
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0x5efa2d4adc0b226b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x304acf0c8d4dcfe8",
#     "start_time": "2024-11-12T11:30:11.430519Z",
#     "end_time": "2024-11-12T11:30:11.430519Z",
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
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0xe4494ab277c5e201",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x304acf0c8d4dcfe8",
#     "start_time": "2024-11-12T11:30:11.430519Z",
#     "end_time": "2024-11-12T11:30:11.431529Z",
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
#     "name": "langchain_openai.chat_models.base.ChatOpenAI",
#     "context": {
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0xaa31db13023b4a9a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x304acf0c8d4dcfe8",
#     "start_time": "2024-11-12T11:30:11.431529Z",
#     "end_time": "2024-11-12T11:30:12.814194Z",
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
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm.gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-11-12T11:30:12.814194Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 75,
#                 "prompt_tokens": 580,
#                 "total_tokens": 655
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
#     "name": "langchain_core.output_parsers.string.StrOutputParser",
#     "context": {
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0x50564d1ca6412e25",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x304acf0c8d4dcfe8",
#     "start_time": "2024-11-12T11:30:12.814194Z",
#     "end_time": "2024-11-12T11:30:12.815228Z",
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
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0x304acf0c8d4dcfe8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa39eef43e3350c89",
#     "start_time": "2024-11-12T11:30:11.429495Z",
#     "end_time": "2024-11-12T11:30:12.815228Z",
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
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0xa39eef43e3350c89",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x4a827696309fdf1c",
#     "start_time": "2024-11-12T11:30:11.427990Z",
#     "end_time": "2024-11-12T11:30:12.815228Z",
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
#         "trace_id": "0x8662e6ad4ce5bb70ce3cfd227b7055af",
#         "span_id": "0x4a827696309fdf1c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-12T11:30:11.003578Z",
#     "end_time": "2024-11-12T11:30:12.815228Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-12T11:30:11.003578Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": []
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-12T11:30:12.815228Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?",
#                 "chat_history": [],
#                 "answer": "Task decomposition is a technique used to break down complex tasks into smaller and simpler steps, making them more manageable for agents or models to handle. This process involves transforming big tasks into multiple smaller tasks to enhance performance and simplify the overall task execution. Task decomposition can be achieved through various methods such as prompting with specific instructions, utilizing human inputs, or leveraging language models with appropriate prompts."
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
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x4ed84af2c0398ac2",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xb3f8473a998c3195",
#     "start_time": "2024-11-12T11:30:12.817748Z",
#     "end_time": "2024-11-12T11:30:12.817748Z",
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
#     "name": "langchain_openai.chat_models.base.ChatOpenAI",
#     "context": {
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x02b8eb0bcf71caa7",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xb3f8473a998c3195",
#     "start_time": "2024-11-12T11:30:12.817748Z",
#     "end_time": "2024-11-12T11:30:13.755816Z",
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
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm.gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-11-12T11:30:13.755816Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 9,
#                 "prompt_tokens": 158,
#                 "total_tokens": 167
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
#     "name": "langchain_core.output_parsers.string.StrOutputParser",
#     "context": {
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x57f456819c17c2d0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xb3f8473a998c3195",
#     "start_time": "2024-11-12T11:30:13.755816Z",
#     "end_time": "2024-11-12T11:30:13.756930Z",
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
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x81bad8adba010b41",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xb3f8473a998c3195",
#     "start_time": "2024-11-12T11:30:13.756930Z",
#     "end_time": "2024-11-12T11:30:14.121735Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "span.type": "retrieval",
#         "entity.count": 2,
#         "entity.1.name": "Chroma",
#         "entity.1.type": "vectorstore.Chroma",
#         "entity.2.name": "text-embedding-ada-002",
#         "entity.2.type": "model.embedding.text-embedding-ada-002"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-12T11:30:13.756930Z",
#             "attributes": {
#                 "question": "What are some common methods for task decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-12T11:30:14.121735Z",
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
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0xb3f8473a998c3195",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x86d1ad418f491ffc",
#     "start_time": "2024-11-12T11:30:12.817748Z",
#     "end_time": "2024-11-12T11:30:14.121735Z",
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
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x86d1ad418f491ffc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe8c14786ebf1413f",
#     "start_time": "2024-11-12T11:30:12.816728Z",
#     "end_time": "2024-11-12T11:30:14.121735Z",
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
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x5be06dffee192e6f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc98bc72da465429a",
#     "start_time": "2024-11-12T11:30:14.123735Z",
#     "end_time": "2024-11-12T11:30:14.124735Z",
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
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x369cdd06639f9c89",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc98bc72da465429a",
#     "start_time": "2024-11-12T11:30:14.124735Z",
#     "end_time": "2024-11-12T11:30:14.125735Z",
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
#     "name": "langchain_openai.chat_models.base.ChatOpenAI",
#     "context": {
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x354428422efd9cab",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc98bc72da465429a",
#     "start_time": "2024-11-12T11:30:14.125735Z",
#     "end_time": "2024-11-12T11:30:15.526970Z",
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
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm.gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "metadata",
#             "timestamp": "2024-11-12T11:30:15.526970Z",
#             "attributes": {
#                 "temperature": 0.7,
#                 "completion_tokens": 77,
#                 "prompt_tokens": 636,
#                 "total_tokens": 713
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
#     "name": "langchain_core.output_parsers.string.StrOutputParser",
#     "context": {
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x4df3e85ebbee40c0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc98bc72da465429a",
#     "start_time": "2024-11-12T11:30:15.526970Z",
#     "end_time": "2024-11-12T11:30:15.526970Z",
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
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0xc98bc72da465429a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x159b720b2f3c750d",
#     "start_time": "2024-11-12T11:30:14.122735Z",
#     "end_time": "2024-11-12T11:30:15.526970Z",
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
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0x159b720b2f3c750d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe8c14786ebf1413f",
#     "start_time": "2024-11-12T11:30:14.122735Z",
#     "end_time": "2024-11-12T11:30:15.526970Z",
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
#         "trace_id": "0x1169d414b97bc4980f6f0295702a47ff",
#         "span_id": "0xe8c14786ebf1413f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-12T11:30:12.815228Z",
#     "end_time": "2024-11-12T11:30:15.526970Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-12T11:30:12.815228Z",
#             "attributes": {
#                 "input": "What are common ways of doing it?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-12T11:30:15.526970Z",
#             "attributes": {
#                 "input": "What are common ways of doing it?",
#                 "answer": "Task decomposition can be commonly done through prompting with language models using simple instructions like \"Steps for XYZ\" or \"What are the subgoals for achieving XYZ?\" Another method is to provide task-specific instructions tailored to the nature of the task, such as \"Write a story outline\" for novel writing. Additionally, task decomposition can involve human inputs to break down complex tasks into more manageable components."
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