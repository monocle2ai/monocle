

import os

import bs4
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
                workflow_name="langchain_app_1",
                span_processors=[BatchSpanProcessor(custom_exporter)],
                wrapper_methods=[])


# llm = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0.7,
# )

@pytest.mark.integration()
def test_langchain_sample(setup):
# llm = OpenAI(model="gpt-3.5-turbo-instruct")
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

    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)

    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if "span.type" in span_attributes and span_attributes["span.type"] == "retrieval":
            # Assertions for all retrieval attributes
            assert span_attributes["entity.1.name"] == "Chroma"
            assert span_attributes["entity.1.type"] == "vectorstore.Chroma"
            assert "entity.1.deployment" in span_attributes
            assert span_attributes["entity.2.name"] == "text-embedding-ada-002"
            assert span_attributes["entity.2.type"] == "model.embedding.text-embedding-ada-002"

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.azure_oai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-3.5-turbo-0125"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-3.5-turbo-0125"


        if not span.parent and span.name == "langchain.workflow":  # Root span
            assert span_attributes["entity.1.name"] == "langchain_app_1"
            assert span_attributes["entity.1.type"] == "workflow.langchain"

# {
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x1889a8cc164c4ac63549b7db2f306836",
#         "span_id": "0xde44999631639bb9",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0d3c9fb243b8f415",
#     "start_time": "2024-11-27T04:16:01.983322Z",
#     "end_time": "2024-11-27T04:16:02.531576Z",
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
#             "timestamp": "2024-11-27T04:16:01.983352Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-27T04:16:02.531546Z",
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
#         "trace_id": "0x1889a8cc164c4ac63549b7db2f306836",
#         "span_id": "0x0d3c9fb243b8f415",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xda093879c3d0bb26",
#     "start_time": "2024-11-27T04:16:01.982513Z",
#     "end_time": "2024-11-27T04:16:02.532364Z",
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
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x1889a8cc164c4ac63549b7db2f306836",
#         "span_id": "0xda093879c3d0bb26",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x719f3fb45d716409",
#     "start_time": "2024-11-27T04:16:01.981783Z",
#     "end_time": "2024-11-27T04:16:02.532935Z",
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
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x1889a8cc164c4ac63549b7db2f306836",
#         "span_id": "0x2c037b5fd754d88c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x719f3fb45d716409",
#     "start_time": "2024-11-27T04:16:02.533215Z",
#     "end_time": "2024-11-27T04:16:02.534542Z",
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
#     "name": "langchain_openai.llms.azure.AzureOpenAI",
#     "context": {
#         "trace_id": "0x1889a8cc164c4ac63549b7db2f306836",
#         "span_id": "0x3dc35627876a83bf",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x719f3fb45d716409",
#     "start_time": "2024-11-27T04:16:02.534775Z",
#     "end_time": "2024-11-27T04:16:04.846864Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.deployment": "kshitiz-gpt",
#         "entity.1.inference_endpoint": "https://okahu-openai-dev.openai.azure.com/",
#         "entity.2.name": "gpt-3.5-turbo-0125",
#         "entity.2.type": "model.llm.gpt-3.5-turbo-0125"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-27T04:16:04.846810Z",
#             "attributes": {
#                 "system": "",
#                 "user": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: What is Task Decomposition? \nContext: Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to \u201cthink step by step\u201d to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model\u2019s thinking process.\n\nTree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like \"Steps for XYZ.\\n1.\", \"What are the subgoals for achieving XYZ?\", (2) by using task-specific instructions; e.g. \"Write a story outline.\" for writing a novel, or (3) with human inputs.\n\nResources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. Reflect on past decisions and strategies to refine your approach.\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.\n\n(3) Task execution: Expert models execute on the specific tasks and log results.\nInstruction:\n\nWith the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user's request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path. \nAnswer:"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-27T04:16:04.846848Z",
#             "attributes": {
#                 "assistant": " \n\nTask Decomposition is a technique that breaks down complex tasks into smaller and simpler steps. This technique is used to make the task more manageable and easier to execute. The process can be done by LLM with simple prompting, task-specific instructions, or with human inputs. The Tree of Thoughts extends the Chain of Thought by exploring multiple reasoning possibilities at each step. \n\nUser Input: What is Task Decomposition?\nTask Planning: Task Decomposition\nModel Selection: None\nTask Execution: Task Decomposition is a technique that breaks down complex tasks into smaller and simpler steps. This technique is used to make the task more manageable and easier to execute. The process can be done by LLM with simple prompting, task-specific instructions, or with human inputs. The Tree of Thoughts extends the Chain of Thought by exploring multiple reasoning possibilities at each step. \n\n<|im_end|>"
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
#         "trace_id": "0x1889a8cc164c4ac63549b7db2f306836",
#         "span_id": "0x42930da0ac6ffc46",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x719f3fb45d716409",
#     "start_time": "2024-11-27T04:16:04.847043Z",
#     "end_time": "2024-11-27T04:16:04.847346Z",
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
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x1889a8cc164c4ac63549b7db2f306836",
#         "span_id": "0x719f3fb45d716409",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-11-27T04:16:01.972044Z",
#     "end_time": "2024-11-27T04:16:04.847421Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.2.0",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-11-27T04:16:01.973158Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-11-27T04:16:04.847412Z",
#             "attributes": {
#                 "response": " \n\nTask Decomposition is a technique that breaks down complex tasks into smaller and simpler steps. This technique is used to make the task more manageable and easier to execute. The process can be done by LLM with simple prompting, task-specific instructions, or with human inputs. The Tree of Thoughts extends the Chain of Thought by exploring multiple reasoning possibilities at each step. \n\nUser Input: What is Task Decomposition?\nTask Planning: Task Decomposition\nModel Selection: None\nTask Execution: Task Decomposition is a technique that breaks down complex tasks into smaller and simpler steps. This technique is used to make the task more manageable and easier to execute. The process can be done by LLM with simple prompting, task-specific instructions, or with human inputs. The Tree of Thoughts extends the Chain of Thought by exploring multiple reasoning possibilities at each step. \n\n<|im_end|>"
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