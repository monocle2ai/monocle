#Continue with code
import logging
import os
import time

import boto3
import pytest
from botocore.exceptions import ClientError
from common.custom_exporter import CustomConsoleSpanExporter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from monocle_apptrace.instrumentation.common.instrumentor import (
    setup_monocle_telemetry,
)
from monocle_apptrace.instrumentation.common.utils import logger
from monocle_apptrace.instrumentation.metamodel.botocore.handlers.botocore_span_handler import (
    BotoCoreSpanHandler,
)
from opensearchpy import RequestsHttpConnection
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from requests_aws4auth import AWS4Auth

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="bedrock_workflow",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[],
            span_handlers={"botocore_handler": BotoCoreSpanHandler()},
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
    custom_exporter.reset()


def test_bedrock_opensearch(setup):
    query = "how?"
    similar_documents = search_similar_documents_opensearch(query)
    produce_llm_response(query, similar_documents)
    time.sleep(5)
    spans = setup.get_captured_spans()

    for span in spans:
        span_attributes = span.attributes
        if 'span.type' in span_attributes and span_attributes["span.type"] == "retrieval":
            # Assertions for all retrieval attributes
            assert span_attributes["entity.1.name"] == "OpenSearchVectorSearch"
            assert span_attributes["entity.1.type"] == "vectorstore.OpenSearchVectorSearch"
            assert "entity.1.deployment" in span_attributes
            assert span_attributes["entity.2.name"] == "amazon.titan-embed-text-v1"
            assert span_attributes["entity.2.type"] == "model.embedding.amazon.titan-embed-text-v1"

        if 'span.type' in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.aws_bedrock"
            if "entity.1.inference_endpoint" in span_attributes.keys():
                assert "entity.1.inference_endpoint" in span_attributes
                assert span_attributes["entity.2.name"] == "anthropic.claude-3-haiku-20240307-v1:0"
                assert span_attributes["entity.2.type"] == "model.llm.anthropic.claude-3-haiku-20240307-v1:0"


                # Assertions for metadata
                span_input, span_output, span_metadata = span.events
                assert "completion_tokens" in span_metadata.attributes
                assert "prompt_tokens" in span_metadata.attributes
                assert "total_tokens" in span_metadata.attributes


def produce_llm_response(query,similar_documents):
    # Bedrock Client Setup
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Set the model ID, e.g., Jurassic-2 Mid.
    #model_id = "ai21.j2-mid-v1"
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"

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
        logger.info(f"The response provided by the endpoint: {response_text}")
        return response_text

    except (ClientError, Exception) as e:
        logger.error(f"Can't invoke '{model_id}'. Reason: {e}")
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
    opensearch_url = os.environ['OPENSEARCH_BEDROCK_ENDPOINT_URL_BOTO']
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
    docs = retriever.invoke(query)
    logger.info(f"Retrieved docs: {docs}")
    return [doc.page_content for doc in docs]


def test_invalid_credentials(setup):
    # Store original session to restore later
    original_session = boto3.DEFAULT_SESSION
    
    try:
        # Force a new session with invalid credentials
        boto3.DEFAULT_SESSION = None
        session = boto3.Session(
            aws_access_key_id='invalid_key',
            aws_secret_access_key='invalid_secret',
            region_name='us-east-1'
        )

        client = session.client("bedrock-runtime")
        query="how?"
        user_message = f' Answer this Query: {query}'
        conversation = [
            {
                "role": "user",
                "content": [{"text": user_message}],
            }
        ]
        response = client.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=conversation,
            inferenceConfig={"maxTokens": 512}
        )
    except ClientError as e:
        logger.error("Authentication error: %s", str(e))
    finally:
        # Restore original session
        boto3.DEFAULT_SESSION = original_session
    
    time.sleep(5)
    spans = setup.get_captured_spans()
    
    # Find inference spans with errors
    error_found = False
    for span in spans:
        if 'span.type' in span.attributes and span.attributes["span.type"] == "inference":
            events = [e for e in span.events if e.name == "data.output"]
            if len(events) > 0 and "error_code" in events[0].attributes:
                assert events[0].attributes["error_code"] == "error"
                # Check for authentication error
                response_text = events[0].attributes.get("response", "")
                assert "UnrecognizedClientException" in response_text or "InvalidSignatureException" in response_text
                error_found = True
                break
    
    # The test should have captured an error
    assert error_found, "Expected to find an inference span with authentication error"

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x0f3fa2aa55641f087a28b74db6e62023",
#         "span_id": "0xa5ec4d3989e36d5a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xaf8bd4bcf7ca055b",
#     "start_time": "2025-03-24T09:21:37.176950Z",
#     "end_time": "2025-03-24T09:21:40.061557Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "retrieval",
#         "entity.1.name": "OpenSearchVectorSearch",
#         "entity.1.type": "vectorstore.OpenSearchVectorSearch",
#         "entity.1.deployment": "https://vvd9mtj8odrs1h09sul4.us-east-1.aoss.amazonaws.com:443",
#         "entity.2.name": "amazon.titan-embed-text-v1",
#         "entity.2.type": "model.embedding.amazon.titan-embed-text-v1",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-03-24T09:21:40.061557Z",
#             "attributes": {
#                 "input": "how?"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-03-24T09:21:40.061557Z",
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
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x0f3fa2aa55641f087a28b74db6e62023",
#         "span_id": "0xaf8bd4bcf7ca055b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-03-24T09:21:37.175457Z",
#     "end_time": "2025-03-24T09:21:40.061557Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "workflow",
#         "entity.1.name": "bedrock_workflow",
#         "entity.1.type": "workflow.langchain",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_workflow"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "botocore.client.BedrockRuntime",
#     "context": {
#         "trace_id": "0x98b169a71915d1cb17271c6fa4a3f4ca",
#         "span_id": "0xf8f491c41b898b85",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x392759cc40f2f51c",
#     "start_time": "2025-03-24T09:21:40.276929Z",
#     "end_time": "2025-03-24T09:21:48.027415Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "inference",
#         "entity.1.type": "inference.aws_sagemaker",
#         "entity.1.inference_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
#         "entity.2.name": "anthropic.claude-v2:1",
#         "entity.2.type": "model.llm.anthropic.claude-v2:1",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-03-24T09:21:48.027415Z",
#             "attributes": {
#                 "input": [
#                     "{'user': 'how?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-03-24T09:21:48.027415Z",
#             "attributes": {
#                 "response": [
#                     "Unfortunately the context provided does not contain enough information to definitively answer the query \"how?\". The query is left unresolved in the passages. \n\nThe closest information is:\n\n1) Liam wonders aloud \"How?\" after receiving a message from Victor Novak about a recent discovery, indicating surprise or confusion over how Novak already knows about the discovery when they \"only just got back\".\n\n2) There is no additional context provided to explain how Novak became aware of their discovery so quickly after they returned. \n\nSo while the context sets up the question of \"how?\" in Liam's mind, it does not actually provide an answer. There simply isn't enough information provided in these passages to determine the answer. The query \"how?\" as posed in the final line is left unresolved."
#                 ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-03-24T09:21:48.027415Z",
#             "attributes": {
#                 "completion_tokens": 170,
#                 "prompt_tokens": 350,
#                 "total_tokens": 520
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
#     "name": "botocore.client.BedrockRuntime",
#     "context": {
#         "trace_id": "0x98b169a71915d1cb17271c6fa4a3f4ca",
#         "span_id": "0x392759cc40f2f51c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-03-24T09:21:40.275930Z",
#     "end_time": "2025-03-24T09:21:48.027415Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "workflow",
#         "entity.1.name": "bedrock_workflow",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_workflow"
#         },
#         "schema_url": ""
#     }
# }