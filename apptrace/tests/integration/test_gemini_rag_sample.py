import logging
import os
import time

import chromadb
import pytest
from chromadb import ClientAPI, Collection
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions
from common.custom_exporter import CustomConsoleSpanExporter
from google import genai
from google.genai import types
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)
COLLECTION_NAME = "coffee"
EMBEDDING_MODEL_ID = "gemini-embedding-001"
INFERENCE_MODEL_ID = "gemini-2.5-flash"

@pytest.fixture(scope="module")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="gemini_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[]
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

# class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
#     def __call__(self, input):
#         client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#         response = client.models.embed_content(
#             model=EMBEDDING_MODEL_ID,
#             contents=input,
#             config=types.EmbedContentConfig(
#                 task_type="retrieval_document",
#                 title="Custom query"
#             )
#         )
#         return [response.embeddings[0].values]

def setup_embedding(chroma_client: ClientAPI) -> Collection:
    chroma_embedding_model = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        model_name=EMBEDDING_MODEL_ID, api_key=os.environ["GEMINI_API_KEY"]
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=chroma_embedding_model
    )
    coffee_file = os.path.join(os.path.dirname(__file__), "coffee.txt")
    with open(coffee_file, "r") as f:
        coffee_docs = [line.strip() for line in f if line.strip()]
    coffee_doc_ids = [str(i) for i in range(len(coffee_docs))]
    collection.add(ids=coffee_doc_ids, documents=coffee_docs)
    return collection

def get_vector_store() -> Collection:
    chroma_client: ClientAPI = chromadb.PersistentClient(path="./vector_db_gemini")
    chroma_client.heartbeat()
    try:
        vector_store = chroma_client.get_collection(name=COLLECTION_NAME)
    except (ValueError, NotFoundError):
        vector_store = setup_embedding(chroma_client)
    return vector_store

def test_gemini_rag_sample(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    question = "what is latte?"
    embedding_response = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=[question],
        config=types.EmbedContentConfig()
    )
    vector = embedding_response.embeddings[0].values

    vector_store: Collection = get_vector_store()
    results = vector_store.query(
        query_embeddings=vector,
        n_results=15,
        include=["documents"]
    )
    context = "\n".join(str(item) for item in results['documents'][0])

    prompt = (
        f"You are a helpful assistant to answer coffee related questions. "
        f"Use the following pieces of retrieved context to answer the question. "
        f"If you don't have the details in the context, say don't know. "
        f"Context: {context}"
        f"Question: {question}"
    )

    answer = client.models.generate_content(
        model=INFERENCE_MODEL_ID,
        contents=prompt
    )
    time.sleep(5)
    logger.info(answer.text)

    found_workflow_span = False
    spans = setup.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            assert span_attributes["entity.1.type"] == "inference.gemini"
            assert "entity.1.inference_endpoint" in span_attributes
            assert "entity.2.name" in span_attributes
            assert "entity.2.type" in span_attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "retrieval":
            assert span_attributes["entity.1.name"] == EMBEDDING_MODEL_ID
            assert span_attributes["entity.1.type"] == "model.embedding."+EMBEDDING_MODEL_ID

            span_input, span_output = span.events
            assert "input" in span_input.attributes or "input" in span_output.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "workflow":
            found_workflow_span = True
    assert found_workflow_span

# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0xf0a5d3ef97f0e57801d81a9feea288af",
#         "span_id": "0x7f384d83a9a880b3",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x379f8b671bb63430",
#     "start_time": "2025-07-14T07:38:28.656221Z",
#     "end_time": "2025-07-14T07:38:29.323842Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_gemini_rag_sample.py:68",
#         "workflow.name": "gemini_app_1",
#         "span.type": "retrieval",
#         "entity.1.name": "models/embedding-001",
#         "entity.1.type": "model.embedding.models/embedding-001",
#         "entity.count": 1
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T07:38:29.323842Z",
#             "attributes": {
#                 "input": "what is latte?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T07:38:29.323842Z",
#             "attributes": {
#                 "response": "[0.030822385, 0.0022213017, -0.02938879, -0.029132029, 0.046188235, 0.034289565, -0.016391452, 0.010671302, 0.012844557, 0.018323725, -0.01602068, 0.0014577437, -0.025379112, -0.019849936, 0.009398521, -0.048684444, -0.001604624, 0.0676821, -0.017330715, -0.012741997, 0.019594207, -0.014156891, 0.009487487, -0.019823808, -0.025522055, 0.0017733825, 0.0026961914, -0.046004914, -0.04614338, 0.018703822, -0.026467038, 0.004239864, -0.019716661, 0.018195797, 0.02940337, -0.027677884, 0.002173902, 0.032266945, 0.007587704, 0.0032095138, 0.021518864, -0.02212691, -0.03481575, 0.043516632, 0.018086787, -0.008174235, -0.002871654, -0.023930524, -0.041198988, -0.079411715, -0.00094901747, -0.007720717, 0.056230277, -0.01304292, 0.0015891107, -0.0021631187, 0.005747304, 0.021858172, -0.043652255, -0.0056065144, 0.008570162, -0.010769728, 0.004910889, 0.021641932, -0.031155825, -0.0024826895, -0.063641824, 0.023217976, 0.042222798, 0.0059243017, -0.05335476, -0.0338251, 0.03111615, -0.00015980858, -0.05405564, -0.014007462, -0.022351053, 0.038792554, 0.033878546, 0.060461227, -0.0050727124, -0.0966267, -0.041551843, -0.0011006342, -0.056223843, -0.007952101, -0.056202494, 0.031135175, -0.043389697, 0.048464864, -0.011360618, -0.0017826303, 0.026390096, -0.0083446, 0.02252682, -0.013174021, -0.046025723, -0.030414518, 0.0069572623, 0.011850064]..."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xf0a5d3ef97f0e57801d81a9feea288af",
#         "span_id": "0x379f8b671bb63430",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-14T07:38:28.654888Z",
#     "end_time": "2025-07-14T07:38:29.323842Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_gemini_rag_sample.py:68",
#         "span.type": "workflow",
#         "entity.1.name": "gemini_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0x5c6026d729eaed505e9b183695150276",
#         "span_id": "0x23edd15ccfc74942",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x68998326efb6ac6d",
#     "start_time": "2025-07-14T07:38:30.200189Z",
#     "end_time": "2025-07-14T07:38:31.338433Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_gemini_rag_sample.py:91",
#         "workflow.name": "gemini_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.gemini",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/",
#         "entity.2.name": "gemini-2.5-flash",
#         "entity.2.type": "model.llm.gemini-2.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-14T07:38:31.338433Z",
#             "attributes": {
#                 "input": [
#                     "{\"user\": \"You are a helpful assistant to answer coffee related questions. Use the following pieces of retrieved context to answer the question. If you don't have the details in the context, say don't know. Context: A latte consists of one or more shots of espresso, served in a glass (or sometimes a cup), into which hot steamed milk is added\\nCoffee is a hot drink made from the roasted and ground seeds (coffee beans) of a tropical shrub\\nAmericano is a type of coffee drink prepared by diluting an espresso shot with hot water at a 1:3 to 1:4 ratio, resulting in a drink that retains the complex flavors of espresso, but in a lighter wayQuestion: what is latte?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-14T07:38:31.338433Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": "{\"model\": \"A latte consists of one or more shots of espresso, served in a glass (or sometimes a cup), into which hot steamed milk is added.\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-14T07:38:31.338433Z",
#             "attributes": {
#                 "completion_tokens": 29,
#                 "prompt_tokens": 148,
#                 "total_tokens": 269
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x5c6026d729eaed505e9b183695150276",
#         "span_id": "0x68998326efb6ac6d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-14T07:38:30.200189Z",
#     "end_time": "2025-07-14T07:38:31.338433Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_gemini_rag_sample.py:91",
#         "span.type": "workflow",
#         "entity.1.name": "gemini_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }