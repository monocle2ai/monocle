from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
setup_monocle_telemetry(
    workflow_name="sagemaker_workflow_1",
    span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
    wrapper_methods=[])

# Continue with your code
from langchain_community.vectorstores import OpenSearchVectorSearch
import boto3
from requests_aws4auth import AWS4Auth

from langchain_community.embeddings import SagemakerEndpointEmbeddings
from opensearchpy import RequestsHttpConnection
import os
import json
from typing import Dict, List

from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

def produce_response(query):
    #similar_documents = search_similar_documents_opensearch(query)
    return produce_llm_response(query)


def produce_llm_response(query):
    client = boto3.client('sagemaker-runtime', region_name='us-east-1')

    endpoint_name = "okahu-sagemaker-rag-qa-ep"  # Your endpoint name.
    content_type = "application/json"  # The MIME type of the input data in the request body.
    accept = "application/json"  # The desired MIME type of the inference in the response.

    data = {
        "context": """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
""" ,
        "question": query
    }

    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept=accept,
        Body=json.dumps(data)
    )

    content = response['Body'].read()

    # Print the content
    response_str = content.decode('utf-8')
    print(f"The response provided by the endpoint: {response_str}")

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


def search_similar_documents_opensearch(query):
    opensearch_url = os.environ['OPENSEARCH_ENDPOINT_URL']
    index_name = "embeddings"  # Your index name
    content_handler = ContentHandler()
    sagemaker_endpoint_embeddings = SagemakerEndpointEmbeddings(endpoint_name="okahu-sagemaker-rag-embedding-ep",
                                                                region_name="us-east-1",
                                                                content_handler=content_handler)
    region = 'us-east-1'
    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    aws_auth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
                        session_token=credentials.token)
    doc_search = OpenSearchVectorSearch(
        opensearch_url=opensearch_url,
        index_name=index_name,
        embedding_function=sagemaker_endpoint_embeddings,
        http_auth=aws_auth,
        use_ssl=True,
        verify_certs=True,
        ssl_assert_hostname=True,
        ssl_show_warn=True,
        connection_class=RequestsHttpConnection
    )
    docs = doc_search.similarity_search(query)
    print(f"Retrieved docs: {docs}")
    return [doc.page_content for doc in docs]


class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["embedding"]


produce_response("hello")

# {
#     "name": "botocore-sagemaker-invoke-endpoint",
#     "context": {
#         "trace_id": "0x74c550d05bd44bd4bc7791230f2838c1",
#         "span_id": "0x12636d1179add9a6",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-12-05T09:42:08.229975Z",
#     "end_time": "2024-12-05T09:42:10.207668Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "entity.1.name": "sagemaker_workflow_1",
#         "entity.1.type": "workflow.generic",
#         "span.type": "inference",
#         "entity.2.type": "inference.aws_sagemaker",
#         "entity.2.inference_endpoint": "https://runtime.sagemaker.us-east-1.amazonaws.com",
#         "entity.3.name": "okahu-sagemaker-rag-qa-ep",
#         "entity.3.type": "model.llm.okahu-sagemaker-rag-qa-ep",
#         "entity.count": 3
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
