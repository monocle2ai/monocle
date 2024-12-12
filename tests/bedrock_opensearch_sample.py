from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentor import set_context_properties
setup_monocle_telemetry(
    workflow_name="bedrock_workflow",
    span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
    wrapper_methods=[]
)
#Continue with code
import boto3
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_aws import BedrockEmbeddings
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from botocore.exceptions import ClientError
import os
#opensearch endpoint url

def produce_response(query):
    similar_documents = search_similar_documents_opensearch(query)
    return produce_llm_response(query, similar_documents)

def produce_llm_response(query,similar_documents):
    # Bedrock Client Setup
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Set the model ID, e.g., Jurassic-2 Mid.
    model_id = "ai21.j2-mid-v1"

    # Start a conversation with the user message.
    #user_message = "Describe the purpose of a 'hello world' program in one line."
    user_message = build_context(similar_documents)
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
    opensearch_url = os.environ['OPENSEARCH_ENDPOINT_URL']
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



produce_response("hello?")
