import os, sys
import time
import warnings, logging
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb import ClientAPI, Collection
from chromadb.errors import InvalidCollectionException
from chromadb.utils import embedding_functions
from openai import OpenAI
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
custom_exporter = CustomConsoleSpanExporter()

COLLECTION_NAME="coffee"
EMBBEDING_MODEL="text-embedding-ada-002"
INFERENCE_MODEL="gpt-4o-mini"
@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
                workflow_name="langchain_app_1",
                span_processors=[BatchSpanProcessor(custom_exporter)],
                wrapper_methods=[])


# Create vectore store and load data
def setup_embedding(chroma_client:ClientAPI) -> Collection:
    chroma_embedding_model = embedding_functions.OpenAIEmbeddingFunction(
                model_name=EMBBEDING_MODEL, api_key=os.environ["OPENAI_API_KEY"])
    coffee_docs = []
    coffee_doc_ids = []
    index = 0
    file_path = os.path.join(os.path.dirname(__file__), "coffee.txt")
    with open(file_path, "r") as f:
        for line in f:
            coffee_docs.append(line.strip())
            coffee_doc_ids.append(str(index))
            index += 1

    #chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=chroma_embedding_model)
    collection.add(ids=coffee_doc_ids, documents=coffee_docs)
    return collection

def get_vector_store() -> Collection:
    chroma_client:ClientAPI = chromadb.PersistentClient(path="./vector_db")
    chroma_client.heartbeat()
    vector_store:Collection = None
    try:
        vector_store = chroma_client.get_collection(name=COLLECTION_NAME)
    except InvalidCollectionException as ex :
        vector_store = setup_embedding(chroma_client)
    except ValueError as ex:
        vector_store = setup_embedding(chroma_client)
    return vector_store

@pytest.mark.integration()
def test_openai_rag_sample(setup):
    question = "what is latte?"
    openai = OpenAI()
    vector = openai.embeddings.create(input = question.split(" "), model=EMBBEDING_MODEL).data[0].embedding

    vector_store:Collection = get_vector_store()

    results=vector_store.query(
        query_embeddings=vector,
        n_results=15,
        include=["documents"]
    )

    context = "\n".join(str(item) for item in results['documents'][0])

#    prompt=f'{res}````who won the award for the original song`'
    response = openai.chat.completions.create(
        model=INFERENCE_MODEL,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant to answer coffee related questions. Use the following pieces of retrieved context to answer the question. If you don't have the details in the context, say don't know. Context: {context}"},
            {"role": "user", "content": question}
        ]
    )
    time.sleep(5)
    print(response)
    print(response.choices[0].message.content)
    found_workflow_span = False
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4o-mini"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4o-mini"

            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "retrieval":
            # Assertions for embedding attributes
            assert span_attributes["entity.1.name"] == "text-embedding-ada-002"
            assert span_attributes["entity.1.type"] == "model.embedding.text-embedding-ada-002"

            span_input, span_output = span.events
            assert "input" in span_input.attributes
            assert "response" in span_output.attributes
            assert question == span_input.attributes['input']
        
        if "span.type" in span_attributes and span_attributes["span.type"] == "workflow":
            found_workflow_span = True
    assert found_workflow_span

# {
#     "name": "openai.resources.embeddings.Embeddings",
#     "context": {
#         "trace_id": "0x4bf452a0386e739ad50dd092a3bdcfec",
#         "span_id": "0xf326cb431a45ff9b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-03-05T11:55:21.675250Z",
#     "end_time": "2025-03-05T11:55:22.410372Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.generic",
#         "span.type": "retrieval",
#         "entity.2.name": "text-embedding-ada-002",
#         "entity.2.type": "model.embedding.text-embedding-ada-002",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-03-05T11:55:22.407373Z",
#             "attributes": {
#                 "input": "what is latte?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-03-05T11:55:22.410372Z",
#             "attributes": {
#                 "response": "index=0, embedding=[-0.005812963470816612, -0.010421467944979668, 0.014182986691594124, -0.009062426..."
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
#     "name": "openai_inference",
#     "context": {
#         "trace_id": "0xa5eb8e9c9a01060e804f0ebf9d0a259b",
#         "span_id": "0x93b0ad65c1293e79",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-03-05T11:55:22.554756Z",
#     "end_time": "2025-03-05T11:55:23.531018Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.generic",
#         "span.type": "inference",
#         "entity.2.type": "inference.azure_openai",
#         "entity.2.provider_name": "api.openai.com",
#         "entity.2.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.3.name": "gpt-4o-mini",
#         "entity.3.type": "model.llm.gpt-4o-mini",
#         "entity.count": 3
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-03-05T11:55:23.531018Z",
#             "attributes": {
#                 "input": [
#                     "{'system': \"You are a helpful assistant to answer coffee related questions. Use the following pieces of retrieved context to answer the question. If you don't have the details in the context, say don't know. Context: Coffee is a hot drink made from the roasted and ground seeds (coffee beans) of a tropical shrub\\nA latte consists of one or more shots of espresso, served in a glass (or sometimes a cup), into which hot steamed milk is added\\nAmericano is a type of coffee drink prepared by diluting an espresso shot with hot water at a 1:3 to 1:4 ratio, resulting in a drink that retains the complex flavors of espresso, but in a lighter way\"}",
#                     "{'user': 'what is latte?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-03-05T11:55:23.531018Z",
#             "attributes": {
#                 "response": "A latte consists of one or more shots of espresso, served in a glass (or sometimes a cup), into which hot steamed milk is added."
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-03-05T11:55:23.531018Z",
#             "attributes": {
#                 "completion_tokens": 30,
#                 "prompt_tokens": 152,
#                 "total_tokens": 182
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