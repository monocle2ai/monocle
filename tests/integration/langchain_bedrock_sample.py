import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_aws import ChatBedrockConverse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from monocle.tests.common.custom_exporter import CustomConsoleSpanExporter
import boto3
import os
os.environ["OPENAI_API_KEY"] = ""


bedrock_runtime_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
custom_exporter = CustomConsoleSpanExporter()
setup_monocle_telemetry(
    workflow_name="bedrock_rag_workflow",
    span_processors=[BatchSpanProcessor(custom_exporter)],
    wrapper_methods=[],
)

llm = ChatBedrockConverse(
    client=bedrock_runtime_client,
    model_id="ai21.jamba-1-5-mini-v1:0",
    temperature=0.1,
)

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

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
)

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

query = "What is Task Decomposition?"
result = rag_chain.invoke(query)

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

        # Assertions for metadata
        span_input, span_output, span_metadata = span.events
        assert "completion_tokens" in span_metadata.attributes
        assert "prompt_tokens" in span_metadata.attributes
        assert "total_tokens" in span_metadata.attributes

    if not span.parent and span.name == "langchain.workflow":  # Root span
        assert span_attributes["entity.1.name"] == "langchain_app_1"
        assert span_attributes["entity.1.type"] == "workflow.langchain"

# {
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x569a7336ee2a2fe52923844a6eedf79d",
#         "span_id": "0x015f4ec4ff5d61f3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xb0809664c643f839",
#     "start_time": "2024-12-06T11:19:39.619246Z",
#     "end_time": "2024-12-06T11:19:40.225971Z",
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
#             "timestamp": "2024-12-06T11:19:39.619269Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-12-06T11:19:40.225945Z",
#             "attributes": {
#                 "response": "Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated ta..."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x569a7336ee2a2fe52923844a6eedf79d",
#         "span_id": "0xb0809664c643f839",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x902f1c282642e02f",
#     "start_time": "2024-12-06T11:19:39.619011Z",
#     "end_time": "2024-12-06T11:19:40.226404Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x569a7336ee2a2fe52923844a6eedf79d",
#         "span_id": "0x902f1c282642e02f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9308293874a886e6",
#     "start_time": "2024-12-06T11:19:39.618541Z",
#     "end_time": "2024-12-06T11:19:40.226591Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x569a7336ee2a2fe52923844a6eedf79d",
#         "span_id": "0x372e5758c82f6b1b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9308293874a886e6",
#     "start_time": "2024-12-06T11:19:40.226714Z",
#     "end_time": "2024-12-06T11:19:40.227314Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain_aws.chat_models.bedrock_converse.ChatBedrockConverse",
#     "context": {
#         "trace_id": "0x569a7336ee2a2fe52923844a6eedf79d",
#         "span_id": "0x0de4a317052e1ea3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9308293874a886e6",
#     "start_time": "2024-12-06T11:19:40.227407Z",
#     "end_time": "2024-12-06T11:19:41.996350Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "span.type": "inference",
#         "entity.count": 2,
#         "entity.1.type": "inference.azure_oai",
#         "entity.1.inference_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
#         "entity.2.name": "ai21.jamba-1-5-mini-v1:0",
#         "entity.2.type": "model.llm.ai21.jamba-1-5-mini-v1:0"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-12-06T11:19:41.996279Z",
#             "attributes": {
#                 "input": [
#                     "{'human': 'You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don\\'t know the answer, just say that you don\\'t know. Use three sentences maximum and keep the answer concise.\\nQuestion: What is Task Decomposition? \\nContext: Fig. 1. Overview of a LLM-powered autonomous agent system.\\nComponent One: Planning#\\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\\nTask Decomposition#\\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to \u201cthink step by step\u201d to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model\u2019s thinking process.\\n\\nTree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\\nTask decomposition can be done (1) by LLM with simple prompting like \"Steps for XYZ.\\\\n1.\", \"What are the subgoals for achieving XYZ?\", (2) by using task-specific instructions; e.g. \"Write a story outline.\" for writing a novel, or (3) with human inputs.\\n\\nResources:\\n1. Internet access for searches and information gathering.\\n2. Long Term memory management.\\n3. GPT-3.5 powered Agents for delegation of simple tasks.\\n4. File output.\\n\\nPerformance Evaluation:\\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\\n2. Constructively self-criticize your big-picture behavior constantly.\\n3. Reflect on past decisions and strategies to refine your approach.\\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.\\n\\n(3) Task execution: Expert models execute on the specific tasks and log results.\\nInstruction:\\n\\nWith the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user\\'s request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path. \\nAnswer:'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-12-06T11:19:41.996316Z",
#             "attributes": {
#                 "response": [
#                     " Task decomposition is the process of breaking down a complex task into smaller, more manageable steps. This can be done using various methods, such as simple prompting, task-specific instructions, or human input. The goal is to make the task more understandable and easier to complete."
#                 ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2024-12-06T11:19:41.996333Z",
#             "attributes": {
#                 "temperature": 0.1,
#                 "completion_tokens": 55,
#                 "prompt_tokens": 640,
#                 "total_tokens": 695
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain_core.output_parsers.string.StrOutputParser",
#     "context": {
#         "trace_id": "0x569a7336ee2a2fe52923844a6eedf79d",
#         "span_id": "0x469ffc2240a50d36",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9308293874a886e6",
#     "start_time": "2024-12-06T11:19:41.996474Z",
#     "end_time": "2024-12-06T11:19:41.996719Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {},
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# },
# {
#     "name": "langchain.workflow",
#     "context": {
#         "trace_id": "0x569a7336ee2a2fe52923844a6eedf79d",
#         "span_id": "0x9308293874a886e6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-12-06T11:19:39.602991Z",
#     "end_time": "2024-12-06T11:19:41.996782Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "workflow",
#         "entity.1.name": "bedrock_rag_workflow",
#         "entity.1.type": "workflow.langchain"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-12-06T11:19:39.604009Z",
#             "attributes": {
#                 "input": "What is Task Decomposition?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-12-06T11:19:41.996776Z",
#             "attributes": {
#                 "response": " Task decomposition is the process of breaking down a complex task into smaller, more manageable steps. This can be done using various methods, such as simple prompting, task-specific instructions, or human input. The goal is to make the task more understandable and easier to complete."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# }
