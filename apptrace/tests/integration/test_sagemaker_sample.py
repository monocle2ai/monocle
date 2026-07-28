# Continue with your code
import json
import logging
import os
import time
from typing import Dict, List

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from common.custom_exporter import CustomConsoleSpanExporter
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain_community.vectorstores import OpenSearchVectorSearch
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.metamodel.botocore.handlers.botocore_span_handler import (
    BotoCoreSpanHandler,
)
from opensearchpy import RequestsHttpConnection
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from requests_aws4auth import AWS4Auth

logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Endpoint names are read from the environment 
QA_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")
EMB_ENDPOINT = os.environ.get("SAGEMAKER_EMB_ENDPOINT_NAME", "")

# The serverless embedding model is 384-dim by minilm-l6-v2, so the OpenSearch index must be created with that dimension.
OPENSEARCH_INDEX = os.environ.get("SAGEMAKER_OPENSEARCH_INDEX", "embeddings-minilm")

# Cold start errors that occur on nearly every invocation.
COLD_START_ERRORS = {"ModelNotReadyException", "ServiceUnavailable", "ThrottlingException"}
COLD_START_ATTEMPTS = 6
COLD_START_DELAY_SECONDS = 15


def sagemaker_runtime_client():
    """One runtime client shared by both endpoints.

    read_timeout is raised well above botocore's default because a cold start can
    keep the request open longer than the default allows.
    """
    return boto3.client(
        "sagemaker-runtime",
        region_name=AWS_REGION,
        config=Config(
            read_timeout=120, connect_timeout=10, retries={"max_attempts": 3, "mode": "standard"}
        ),
    )


def invoke_with_cold_start_retry(invoke):
    """Retry only the errors a cold serverless endpoint raises, and say so."""
    for attempt in range(COLD_START_ATTEMPTS):
        try:
            return invoke()
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code not in COLD_START_ERRORS or attempt == COLD_START_ATTEMPTS - 1:
                raise
            logger.info(
                "endpoint cold (%s), retrying in %ss", code, COLD_START_DELAY_SECONDS
            )
            time.sleep(COLD_START_DELAY_SECONDS)
    raise RuntimeError("unreachable")


def warm_up_endpoint(runtime, endpoint_name, body):
    """Absorb the cold start before any traced work happens.

    The retriever runs inside LangChain, where individual calls cannot be wrapped,
    so both endpoints are woken first. Spans from these calls are discarded by the
    caller -- otherwise the extra inference spans break the assertions below.
    """
    invoke_with_cold_start_retry(
        lambda: runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(body).encode("utf-8"),
        )
    )
@pytest.fixture(scope="function")
def setup():
    instrumentor = None
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="sagemaker_workflow_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[],
            span_handlers={"botocore_handler":BotoCoreSpanHandler()}
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_sagemaker_sample(setup):
    if not QA_ENDPOINT or not EMB_ENDPOINT:
        logger.error(
            "SAGEMAKER_ENDPOINT_NAME and SAGEMAKER_EMB_ENDPOINT_NAME must be set in the environment"
        )

    query = "hello"
    runtime = sagemaker_runtime_client()

    # Wake both endpoints, then drop the resulting spans so only the traced
    # workflow below is asserted on.
    warm_up_endpoint(runtime, EMB_ENDPOINT, {"text_inputs": ["warm up"]})
    warm_up_endpoint(runtime, QA_ENDPOINT, {"context": "warm up", "question": "what?"})
    setup.reset()

    similar_documents = search_similar_documents_opensearch(query, runtime)
    produce_llm_response(query, similar_documents, runtime)

    time.sleep(5)
    spans = setup.get_captured_spans()

    for span in spans:
        span_attributes = span.attributes
        if 'span.type' in span_attributes and span_attributes["span.type"] == "retrieval":
            # Assertions for all retrieval attributes
            assert span_attributes["entity.1.name"] == "OpenSearchVectorSearch"
            assert span_attributes["entity.1.type"] == "vectorstore.OpenSearchVectorSearch"
            assert "entity.1.deployment" in span_attributes
            assert span_attributes["entity.2.name"] == EMB_ENDPOINT
            assert span_attributes["entity.2.type"] == f"model.embedding.{EMB_ENDPOINT}"

        if 'span.type' in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.aws_sagemaker"
            assert "entity.1.inference_endpoint" in span_attributes
            span_input, span_output,span_metadata = span.events
            if 'input' in span_input.attributes:
                logger.info("Comparing inputs for inference span")
                assert span_attributes["entity.2.name"] == QA_ENDPOINT
                assert span_attributes["entity.2.type"] == f"model.llm.{QA_ENDPOINT}"


def produce_llm_response(query, similar_documents, client):
    content_type = "application/json"  # The MIME type of the input data in the request body.
    accept = "application/json"  # The desired MIME type of the inference in the response.
    context = build_context(similar_documents)
    #user_message = f'Context - {context}\nBased on the above context, answer this Query: {query}'
    data = {
        "context": context,
        "question": query
    }

    response = invoke_with_cold_start_retry(
        lambda: client.invoke_endpoint(
            EndpointName=QA_ENDPOINT,
            ContentType=content_type,
            Accept=accept,
            Body=json.dumps(data)
        )
    )

    content = response['Body'].read()

    response_str = content.decode('utf-8')
    logger.info(f"The response provided by the endpoint: {response_str}")

    answer = json.loads(response_str)["answer"]
    return answer


def build_context(similar_documents):
    if len(similar_documents) > 0:
        documents_concatenated = "-------------END OF DOCUMENT-------------".join(similar_documents)
        return f"""Based on embedding lookup, we've found these documents to be the most relevant from the knowledge 
        base: {documents_concatenated}"""
    else:
        return "We couldn't locate any documents that would be relevant for this question. Please apologize politely " \
               "and say that you don't know the answer if this is not something you can answer on your own."


def search_similar_documents_opensearch(query, client):
    opensearch_url = os.environ['OPENSEARCH_ENDPOINT_URL_BOTO_SAGEMAKER']  # Your OpenSearch endpoint URL
    content_handler = ContentHandler()
    # Reuse the caller's client so the retriever gets the same timeout and retry
    # settings as the direct invocation.
    sagemaker_endpoint_embeddings = SagemakerEndpointEmbeddings(endpoint_name=EMB_ENDPOINT,
                                                                client=client,
                                                                content_handler=content_handler)
    service = 'aoss'
    # credentials = boto3.Session().get_credentials()
    # aws_auth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
    #                     session_token=credentials.token)
    aws_auth = AWS4Auth(os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"], AWS_REGION, service,
                        )
    doc_search = OpenSearchVectorSearch(
        opensearch_url=opensearch_url,
        index_name=OPENSEARCH_INDEX,
        embedding_function=sagemaker_endpoint_embeddings,
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


class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        try:
            data = output.read()
        except Exception:
            if hasattr(output, '_raw_stream') and hasattr(output._raw_stream, 'data'):
                data = output._raw_stream.data
            else:
                raise
        
        response_json = json.loads(data.decode("utf-8"))
        return response_json["embedding"]

# {
#     "name": "botocore.client.SageMakerRuntime",
#     "context": {
#         "trace_id": "0x6b4c838381c3f335e34489c8bfa15876",
#         "span_id": "0x4176098d5f9f4133",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xbd94a5f60b616d3e",
#     "start_time": "2025-03-24T10:05:58.868702Z",
#     "end_time": "2025-03-24T10:06:00.664215Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "inference",
#         "entity.1.type": "inference.aws_sagemaker",
#         "entity.1.inference_endpoint": "https://runtime.sagemaker.us-east-1.amazonaws.com",
#         "entity.2.name": "okahu-sagemaker-rag-embedding-ep",
#         "entity.2.type": "model.llm.okahu-sagemaker-rag-embedding-ep",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-03-24T10:06:00.196440Z",
#             "attributes": {
#                 "input": []
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-03-24T10:06:00.664215Z",
#             "attributes": {
#                 "response": []
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-03-24T10:06:00.664215Z",
#             "attributes": {}
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "sagemaker_workflow_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x6b4c838381c3f335e34489c8bfa15876",
#         "span_id": "0xbd94a5f60b616d3e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x866b0fb2d9d6a780",
#     "start_time": "2025-03-24T10:05:58.865659Z",
#     "end_time": "2025-03-24T10:06:03.581234Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "retrieval",
#         "entity.1.name": "OpenSearchVectorSearch",
#         "entity.1.type": "vectorstore.OpenSearchVectorSearch",
#         "entity.1.deployment": "https://vvd9mtj8odrs1h09sul4.us-east-1.aoss.amazonaws.com:443",
#         "entity.2.name": "okahu-sagemaker-rag-embedding-ep",
#         "entity.2.type": "model.embedding.okahu-sagemaker-rag-embedding-ep",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-03-24T10:06:03.581234Z",
#             "attributes": {
#                 "input": "hello"
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-03-24T10:06:03.581234Z",
#             "attributes": {
#                 "response": "\n\n      LLM Powered Autonomous Agents\n    \nDate: June 23, 2023  |  Estimated Reading Time: 31 min  |..."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "sagemaker_workflow_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.vectorstores.base.VectorStoreRetriever",
#     "context": {
#         "trace_id": "0x6b4c838381c3f335e34489c8bfa15876",
#         "span_id": "0x866b0fb2d9d6a780",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-03-24T10:05:58.864662Z",
#     "end_time": "2025-03-24T10:06:03.581234Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "workflow",
#         "entity.1.name": "sagemaker_workflow_1",
#         "entity.1.type": "workflow.langchain",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "sagemaker_workflow_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "botocore.client.SageMakerRuntime",
#     "context": {
#         "trace_id": "0x88dda10e8a6239cc0cab9de79225839e",
#         "span_id": "0x632487e1115f81ca",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xca25371cbee68d24",
#     "start_time": "2025-03-24T10:06:03.922480Z",
#     "end_time": "2025-03-24T10:06:06.156310Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "inference",
#         "entity.1.type": "inference.aws_sagemaker",
#         "entity.1.inference_endpoint": "https://runtime.sagemaker.us-east-1.amazonaws.com",
#         "entity.2.name": "okahu-sagemaker-rag-qa-ep",
#         "entity.2.type": "model.llm.okahu-sagemaker-rag-qa-ep",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-03-24T10:06:06.156310Z",
#             "attributes": {
#                 "input": [
#                     "hello"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-03-24T10:06:06.156310Z",
#             "attributes": {
#                 "response": [
#                     "June 23, 2023"
#                 ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-03-24T10:06:06.156310Z",
#             "attributes": {}
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "sagemaker_workflow_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "botocore.client.SageMakerRuntime",
#     "context": {
#         "trace_id": "0x88dda10e8a6239cc0cab9de79225839e",
#         "span_id": "0xca25371cbee68d24",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-03-24T10:06:03.921480Z",
#     "end_time": "2025-03-24T10:06:06.156310Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "workflow",
#         "entity.1.name": "sagemaker_workflow_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "sagemaker_workflow_1"
#         },
#         "schema_url": ""
#     }
# }
