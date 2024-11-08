

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAI, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
import os
os.environ["AZURE_OPENAI_API_DEPLOYMENT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_API_VERSION"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["OPENAI_API_KEY"] = ""
setup_monocle_telemetry(
            workflow_name="langchain_app_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[])

# llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
llm = AzureOpenAI(
    # engine=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
    azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    temperature=0.1,
    # model="gpt-4",

    model="gpt-3.5-turbo-0125")
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
#         "trace_id": "0xee531670266befa8e3bd5dcf31d2a08b",
#         "span_id": "0xfa3ef134b3368f45",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x45b0408368897207",
#     "start_time": "2024-10-30T09:21:23.642049Z",
#     "end_time": "2024-10-30T09:21:24.347534Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
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
#             "timestamp": "2024-10-30T09:21:23.642167Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-30T09:21:24.347519Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xee531670266befa8e3bd5dcf31d2a08b",
#         "span_id": "0x45b0408368897207",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xcbdb94928dc3340d",
#     "start_time": "2024-10-30T09:21:23.641702Z",
#     "end_time": "2024-10-30T09:21:24.347840Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xee531670266befa8e3bd5dcf31d2a08b",
#         "span_id": "0xcbdb94928dc3340d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xecd1a8a19417dc8e",
#     "start_time": "2024-10-30T09:21:23.641252Z",
#     "end_time": "2024-10-30T09:21:24.348115Z",
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
# },
# {
#     "name": "langchain.task.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0xee531670266befa8e3bd5dcf31d2a08b",
#         "span_id": "0x9a9cf227a70702a6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xecd1a8a19417dc8e",
#     "start_time": "2024-10-30T09:21:24.348227Z",
#     "end_time": "2024-10-30T09:21:24.348663Z",
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
# },
# {
#     "name": "langchain.task.AzureOpenAI",
#     "context": {
#         "trace_id": "0xee531670266befa8e3bd5dcf31d2a08b",
#         "span_id": "0x2f1a872aa6f80dce",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xecd1a8a19417dc8e",
#     "start_time": "2024-10-30T09:21:24.348733Z",
#     "end_time": "2024-10-30T09:21:26.603370Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.provider_name": "okahu-openai-dev.openai.azure.com",
#         "entity.1.deployment": "kshitiz-gpt",
#         "entity.1.inference_endpoint": "https://okahu-openai-dev.openai.azure.com/",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm.gpt-3.5-turbo-0125"
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
#         "trace_id": "0xee531670266befa8e3bd5dcf31d2a08b",
#         "span_id": "0x8f219ad1d33dd447",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xecd1a8a19417dc8e",
#     "start_time": "2024-10-30T09:21:26.603643Z",
#     "end_time": "2024-10-30T09:21:26.604075Z",
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
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0xee531670266befa8e3bd5dcf31d2a08b",
#         "span_id": "0xecd1a8a19417dc8e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-10-30T09:21:23.636838Z",
#     "end_time": "2024-10-30T09:21:26.604151Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-10-30T09:21:23.636873Z",
#             "attributes": {
#                 "question": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-10-30T09:21:26.604134Z",
#             "attributes": {
#                 "response": " \n\nTask decomposition is a technique that breaks down complex tasks into smaller and simpler steps. It can be done by LLM with simple prompting, task-specific instructions, or human inputs. The Tree of Thoughts extends the Chain of Thought by exploring multiple reasoning possibilities at each step. I used the Chain of Thought to decompose the task into smaller steps and then used the LLM to execute the task. The results are logged in the file output. The file path is {{ file_path }}.<|im_end|>"
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