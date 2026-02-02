
import logging
import time
import boto3
import bs4
import pytest
from langsmith import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrockConverse
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.metamodel.botocore.handlers.botocore_span_handler import (
    BotoCoreSpanHandler,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
    memory_exporter = InMemorySpanExporter()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="bedrock_rag_workflow",
            span_processors=[SimpleSpanProcessor(memory_exporter)],
            wrapper_methods=[],
            span_handlers={"botocore_handler": BotoCoreSpanHandler()},
        )
        yield memory_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_langchain_bedrock_sample(setup):
    bedrock_runtime_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
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

    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt")

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

    logger.info(result)
    time.sleep(5)  # Allow time for spans to be exported

    verify_spans(memory_exporter=setup)


def verify_spans(memory_exporter):
    """Verify all spans are generated correctly for the RAG workflow."""
    spans = memory_exporter.get_finished_spans()
    logger.info(f"Captured {len(spans)} spans")
    assert len(spans) > 0, "No spans captured"

    # Find workflow span
    workflow_span = None
    for span in spans:
        if span.attributes.get("span.type") == "workflow":
            workflow_span = span
            break
    assert workflow_span is not None, "Expected to find workflow span"
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "bedrock_rag_workflow"
    assert workflow_span.attributes["entity.1.type"] == "workflow.openai"

    # Find inference spans ("inference" or "inference.framework")
    inference_spans = [s for s in spans if s.attributes.get("span.type") in ("inference", "inference.framework")]
    assert len(inference_spans) > 0, "Expected to find at least one inference span"
    # Only one LLM call expected
    assert len(inference_spans) == 1, "Expected exactly one inference span for the LLM call"
    inf_span = inference_spans[0]
    # Check key attributes
    assert inf_span.attributes.get("entity.1.type") == "inference.aws_bedrock"
    assert inf_span.attributes.get("entity.2.name") == "ai21.jamba-1-5-mini-v1:0"
    assert inf_span.attributes.get("entity.2.type") == "model.llm.ai21.jamba-1-5-mini-v1:0"
    assert inf_span.attributes.get("entity.1.inference_endpoint") == "https://bedrock-runtime.us-east-1.amazonaws.com"
    # Check events
    events = inf_span.events
    assert len(events) == 3, "Expected 3 events on inference span (input, output, metadata)"
    event_names = [e.name for e in events]
    assert "data.input" in event_names
    assert "data.output" in event_names
    assert "metadata" in event_names
    # Check metadata event attributes
    metadata_event = next(e for e in events if e.name == "metadata")
    for k in ["completion_tokens", "prompt_tokens", "total_tokens"]:
        assert k in metadata_event.attributes and isinstance(metadata_event.attributes[k], int)

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "openai.resources.embeddings.Embeddings",
#     "context": {
#         "trace_id": "0xf9e38c85078e5b950bb803ce39b221d7",
#         "span_id": "0x70ad4d0b51b85a09",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7196d610a49e14bf",
#     "start_time": "2025-07-02T20:46:37.723052Z",
#     "end_time": "2025-07-02T20:46:40.277692Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_community/embeddings/openai.py:120",
#         "workflow.name": "bedrock_rag_workflow",
#         "span.type": "embedding",
#         "entity.1.name": "text-embedding-ada-002",
#         "entity.1.type": "model.embedding.text-embedding-ada-002",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-02T20:46:40.232498Z",
#             "attributes": {}
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T20:46:40.277650Z",
#             "attributes": {
#                 "response": "index=0, embedding=[0.0040525333024561405, 0.008230944164097309, -0.0056085106916725636, -0.02394456..."
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
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xf9e38c85078e5b950bb803ce39b221d7",
#         "span_id": "0x7196d610a49e14bf",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T20:46:37.722990Z",
#     "end_time": "2025-07-02T20:46:40.277748Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_community/embeddings/openai.py:120",
#         "span.type": "workflow",
#         "entity.1.name": "bedrock_rag_workflow",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.embeddings.Embeddings",
#     "context": {
#         "trace_id": "0xc0f52f6b216e4e31b12e87005f38cf04",
#         "span_id": "0x32fcd2b64b94c825",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xb0fc2f8aa76093bf",
#     "start_time": "2025-07-02T20:46:41.153778Z",
#     "end_time": "2025-07-02T20:46:41.955336Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_community/embeddings/openai.py:120",
#         "workflow.name": "bedrock_rag_workflow",
#         "span.type": "embedding.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.runnables.base.RunnableSequence",
#     "context": {
#         "trace_id": "0xc0f52f6b216e4e31b12e87005f38cf04",
#         "span_id": "0x62935097e25127c5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x26d2972131a15476",
#     "start_time": "2025-07-02T20:46:41.151020Z",
#     "end_time": "2025-07-02T20:46:41.968427Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3758",
#         "workflow.name": "bedrock_rag_workflow",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0xc0f52f6b216e4e31b12e87005f38cf04",
#         "span_id": "0xf24f4a421569ae63",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x26d2972131a15476",
#     "start_time": "2025-07-02T20:46:41.969765Z",
#     "end_time": "2025-07-02T20:46:41.970744Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "bedrock_rag_workflow",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_aws.chat_models.bedrock_converse.ChatBedrockConverse",
#     "context": {
#         "trace_id": "0xc0f52f6b216e4e31b12e87005f38cf04",
#         "span_id": "0xd87f68bdfcda7ab7",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x26d2972131a15476",
#     "start_time": "2025-07-02T20:46:41.971474Z",
#     "end_time": "2025-07-02T20:46:44.524823Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "bedrock_rag_workflow",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.aws_bedrock",
#         "entity.1.inference_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
#         "entity.2.name": "ai21.jamba-1-5-mini-v1:0",
#         "entity.2.type": "model.llm.ai21.jamba-1-5-mini-v1:0",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-02T20:46:44.524669Z",
#             "attributes": {
#                 "input": [
#                     "{'human': 'You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don\\'t know the answer, just say that you don\\'t know. Use three sentences maximum and keep the answer concise.\\nQuestion: What is Task Decomposition? \\nContext: Component One: Planning#\\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\\nTask Decomposition#\\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to \u201cthink step by step\u201d to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model\u2019s thinking process.\\nTree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\\n\\nTask decomposition can be done (1) by LLM with simple prompting like \"Steps for XYZ.\\\\n1.\", \"What are the subgoals for achieving XYZ?\", (2) by using task-specific instructions; e.g. \"Write a story outline.\" for writing a novel, or (3) with human inputs.\\nAnother quite distinct approach, LLM+P (Liu et al. 2023), involves relying on an external classical planner to do long-horizon planning. This approach utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface to describe the planning problem. In this process, LLM (1) translates the problem into \u201cProblem PDDL\u201d, then (2) requests a classical planner to generate a PDDL plan based on an existing \u201cDomain PDDL\u201d, and finally (3) translates the PDDL plan back into natural language. Essentially, the planning step is outsourced to an external tool, assuming the availability of domain-specific PDDL and a suitable planner which is common in certain robotic setups but not in many other domains.\\nSelf-Reflection#\\n\\nIllustration of how HuggingGPT works. (Image source: Shen et al. 2023)\\n\\nThe system comprises of 4 stages:\\n(1) Task planning: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, dependencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.\\nInstruction:\\n\\nResources:\\n1. Internet access for searches and information gathering.\\n2. Long Term memory management.\\n3. GPT-3.5 powered Agents for delegation of simple tasks.\\n4. File output.\\n\\nPerformance Evaluation:\\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\\n2. Constructively self-criticize your big-picture behavior constantly.\\n3. Reflect on past decisions and strategies to refine your approach.\\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps. \\nAnswer:'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T20:46:44.524755Z",
#             "attributes": {
#                 "response": "{'ai': ' Task decomposition is the process of breaking down a complex task into smaller, more manageable steps. This can be done using various methods, such as using large language models (LLMs) with simple prompts, task-specific instructions, or even human inputs. Another approach involves using an external classical planner to perform long-horizon planning, which utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface.'}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T20:46:44.524798Z",
#             "attributes": {
#                 "temperature": 0.1,
#                 "completion_tokens": 83,
#                 "prompt_tokens": 749,
#                 "total_tokens": 832
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
#  Task decomposition is the process of breaking down a complex task into smaller, more manageable steps. This can be done using various methods, such as using large language models (LLMs) with simple prompts, task-specific instructions, or even human inputs. Another approach involves using an external classical planner to perform long-horizon planning, which utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface.{
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xc0f52f6b216e4e31b12e87005f38cf04",
#         "span_id": "0x8637184ba1a44201",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T20:46:41.135299Z",
#     "end_time": "2025-07-02T20:46:44.527171Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_bedrock_sample.py:75",
#         "span.type": "workflow",
#         "entity.1.name": "bedrock_rag_workflow",
#         "entity.1.type": "workflow.langchain",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_rag_workflow"
#         },
#         "schema_url": ""
#     }
# }
