import os
import time

import boto3
import pytest
from botocore.exceptions import ClientError
from common.custom_exporter import CustomConsoleSpanExporter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from requests_aws4auth import AWS4Auth

from monocle_apptrace.instrumentation.common.instrumentor import (
    set_context_properties,
    setup_monocle_telemetry,
)

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="bedrock_workflow",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

@pytest.mark.integration()
def test_bedrock_opensearch(setup):
    query = "how?"
    similar_documents = search_similar_documents_opensearch(query)
    produce_llm_response(query, similar_documents)
    time.sleep(20)
    spans = custom_exporter.get_captured_spans()

    for span in spans:
        span_attributes = span.attributes
        if 'span.type' in span_attributes and span_attributes["span.type"] == "retrieval":
            # Assertions for all retrieval attributes
            assert span_attributes["entity.1.name"] == "bedrock_workflow"
            assert span_attributes["entity.1.type"] == "workflow.langchain"
            assert span_attributes["entity.2.name"] == "OpenSearchVectorSearch"
            assert span_attributes["entity.2.type"] == "vectorstore.OpenSearchVectorSearch"
            assert "entity.2.deployment" in span_attributes
            assert span_attributes["entity.3.name"] == "amazon.titan-embed-text-v1"
            assert span_attributes["entity.3.type"] == "model.embedding.amazon.titan-embed-text-v1"

        if 'span.type' in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.name"] == "bedrock_workflow"
            assert span_attributes["entity.1.type"] == "workflow.generic"
            assert span_attributes["entity.2.type"] == "inference.aws_bedrock"
            assert "entity.2.inference_endpoint" in span_attributes
            assert span_attributes["entity.3.name"] == "ai21.j2-mid-v1"
            assert span_attributes["entity.3.type"] == "model.llm.ai21.j2-mid-v1"


            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes


def produce_llm_response(query,similar_documents):
    # Bedrock Client Setup
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Set the model ID, e.g., Jurassic-2 Mid.
    model_id = "ai21.j2-mid-v1"

    context = build_context(similar_documents)
    user_message = f'Context - {context}\nBased on the above context, answer this Query: {query}'
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
        )

        response_text = response["output"]["message"]["content"][0]["text"]
        print(f"The response provided by the endpoint: {response_text}")
        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)


def build_context(similar_documents):
    if similar_documents:
        documents_concatenated = "-------------END OF DOCUMENT-------------".join(similar_documents)
        return f"""Based on embedding lookup, we've found these documents to be the most relevant from the knowledge
        base: {documents_concatenated}"""
    else:
        return "We couldn't locate any documents that would be relevant for this question. Please apologize politely " \
               "and say that you don't know the answer if this is not something you can answer on your own."

def search_similar_documents_opensearch(query):
    opensearch_url = os.environ['OPENSEARCH_ENDPOINT_URL_BOTO']
    index_name = "embeddings-bedrock"  # Your index name

    bedrock_embeddings = BedrockEmbeddings(region_name="us-east-1",model_id="amazon.titan-embed-text-v1")
    region = 'us-east-1'
    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    aws_auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

    doc_search = OpenSearchVectorSearch(
        opensearch_url=opensearch_url,
        index_name=index_name,
        embedding_function= bedrock_embeddings,
        http_auth=aws_auth,
        use_ssl=True,
        verify_certs=True,
        ssl_assert_hostname=True,
        ssl_show_warn=True,
        connection_class=RequestsHttpConnection
    )
    retriever = doc_search.as_retriever()
    docs = retriever.get_relevant_documents(query)
    print(f"Retrieved docs: {docs}")
    return [doc.page_content for doc in docs]

# {
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0xa98d7934d334d3afd58b34fa4b24862f",
#         "span_id": "0x0a52d9a5c65b8b80",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-12-26T10:33:43.036674Z",
#     "end_time": "2024-12-26T10:33:45.965446Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "entity.1.name": "bedrock_workflow",
#         "entity.1.type": "workflow.langchain",
#         "span.type": "retrieval",
#         "entity.2.name": "OpenSearchVectorSearch",
#         "entity.2.type": "vectorstore.OpenSearchVectorSearch",
#         "entity.2.deployment": "https://vvd9mtj8odrs1h09sul4.us-east-1.aoss.amazonaws.com:443",
#         "entity.3.name": "amazon.titan-embed-text-v1",
#         "entity.3.type": "model.embedding.amazon.titan-embed-text-v1",
#         "entity.count": 3
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-12-26T10:33:45.965446Z",
#             "attributes": {
#                 "input": "how?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-12-26T10:33:45.965446Z",
#             "attributes": {
#                 "response": "\"How?\" Liam wondered aloud. \"We only just got back.\"\n\nTheir concern grew when a black SUV pulled up ..."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_workflow"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "botocore-bedrock-runtime-invoke-endpoint",
#     "context": {
#         "trace_id": "0x0ace7bfa12f8161f2e4460471f912090",
#         "span_id": "0x2c5cc59353524cbc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-12-26T10:33:46.090828Z",
#     "end_time": "2024-12-26T10:33:47.115262Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "entity.1.name": "bedrock_workflow",
#         "entity.1.type": "workflow.generic",
#         "span.type": "inference",
#         "entity.2.type": "inference.aws_bedrock",
#         "entity.2.inference_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
#         "entity.3.name": "ai21.j2-mid-v1",
#         "entity.3.type": "model.llm.ai21.j2-mid-v1",
#         "entity.count": 3
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2024-12-26T10:33:47.115262Z",
#             "attributes": {
#                 "input": [
#                     "{'user': 'how?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2024-12-26T10:33:47.115262Z",
#             "attributes": {
#                 "response": [
#                     "\nLiam wondered how Victor Novak already knew about their discovery, even though they had just returned."
#                 ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2024-12-26T10:33:47.115262Z",
#             "attributes": {
#                 "completion_tokens": 14,
#                 "prompt_tokens": 299,
#                 "total_tokens": 313
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_workflow"
#         },
#         "schema_url": ""
#     }
# }