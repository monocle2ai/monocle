import os
import time
import chromadb
from chromadb import ClientAPI, Collection
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

custom_exporter = CustomConsoleSpanExporter()

COLLECTION_NAME = "coffee"
EMBEDDING_MODEL_ID = "models/embedding-001"
INFERENCE_MODEL_ID = "gemini-2.5-flash"

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="gemini_app_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

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
    except ValueError:
        vector_store = setup_embedding(chroma_client)
    return vector_store

@pytest.mark.integration()
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
    print(answer.text)

    found_workflow_span = False
    spans = custom_exporter.get_captured_spans()
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