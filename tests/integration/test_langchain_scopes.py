

import os

import bs4
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.chain_exec import TestScopes
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, start_scope, stop_scope, monocle_trace_scope

CHAT_SCOPE_NAME = "chat"
custom_exporter = CustomConsoleSpanExporter()
@pytest.fixture(scope="module")
def setup():
      os.environ["HAYSTACK_AUTO_TRACE_ENABLED"] = "False"
      setup_monocle_telemetry(
                workflow_name="langchain_app_1",
                span_processors=[SimpleSpanProcessor(custom_exporter)],
                wrapper_methods=[])

# llm = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0.7,
# )
@staticmethod
def setup_chain():
    """ Setup the langchain chain for the test """
    llm = AzureChatOpenAI(
        # engine=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        temperature=0.1,
        # model="gpt-4",

        model="gpt-3.5-turbo-0125")
    # Load, chunk and index the contents of the blog.
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
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
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
    return rag_chain


@pytest.mark.integration()
def test_scope_api(setup):
    """ Test setting scope via start/stop API. Verify that the scope is effective across chains/traces, and not in effect after stop is called"""
    rag_chain = setup_chain()
    scope_name = "message"
    token = start_scope(scope_name)

    # 1st chain run
    custom_exporter.reset()
    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)
    message_scope_id = None
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)
    custom_exporter.reset()

    # 2nd chain run
    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        assert span_attributes.get("scope."+scope_name) == message_scope_id
    custom_exporter.reset()

    stop_scope(scope_name, token)
    
    # 3rd chain run
    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        assert span_attributes.get("scope."+scope_name) is None

@pytest.mark.integration()
def test_scope_api_with_value(setup):
    """ Test setting scope via start/stop API with specific scope value """
    rag_chain = setup_chain()
    scope_name = "dummy"
    scope_value = "test123"
    token = start_scope(scope_name, scope_value)

    custom_exporter.reset()
    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)
    message_scope_id = None
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)
        assert message_scope_id == scope_value
    stop_scope(scope_name, token)

@monocle_trace_scope(scope_name=CHAT_SCOPE_NAME)
def run_chain_with_scope(chain, message):
    result = chain.invoke(message)
    print(result)
    return result

@pytest.mark.integration()
def test_scope_wrapper(setup):
    """ Test setting scope at function level using decorator """
    custom_exporter.reset()
    rag_chain = setup_chain()
    scope_name = CHAT_SCOPE_NAME
    result = run_chain_with_scope(rag_chain, "What is Task Decomposition?")
    print(result)
    spans = custom_exporter.get_captured_spans()
    message_scope_id = None
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)

@pytest.mark.integration()
def test_scope_config(setup):
    """ Test setting scope at function level using external configuartion """
    custom_exporter.reset()
    test_scope = TestScopes()
    chain = setup_chain()
    result = test_scope.config_scope_func(chain, "What is Task Decomposition?")
    print(result)
    scope_name = "question"
    spans = custom_exporter.get_captured_spans()
    message_scope_id = None
    for span in spans:
        span_attributes = span.attributes
        if message_scope_id is None:
            message_scope_id = span_attributes.get("scope."+scope_name)
            assert message_scope_id is not None
        else:
            assert message_scope_id == span_attributes.get("scope."+scope_name)
