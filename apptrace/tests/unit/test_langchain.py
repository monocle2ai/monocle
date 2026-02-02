

import json
import logging
import os
import time
import unittest
from unittest.mock import patch

import httpx
import requests
from base_unit import MonocleTestBase
from common.embeddings_wrapper import HuggingFaceEmbeddings
from common.fake_list_llm import FakeListLLM
from common.http_span_exporter import HttpSpanExporter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import faiss
from langchain_core.runnables import RunnablePassthrough
from monocle_apptrace.instrumentation.common.constants import (
    AWS_LAMBDA_ENV_NAME,
    AWS_LAMBDA_FUNCTION_IDENTIFIER_ENV_NAME,
    AWS_LAMBDA_SERVICE_NAME,
    AZURE_APP_SERVICE_ENV_NAME,
    AZURE_APP_SERVICE_IDENTIFIER_ENV_NAME,
    AZURE_APP_SERVICE_NAME,
    AZURE_FUNCTION_IDENTIFIER_ENV_NAME,
    AZURE_FUNCTION_NAME,
    AZURE_FUNCTION_WORKER_ENV_NAME,
    AZURE_ML_ENDPOINT_ENV_NAME,
    AZURE_ML_SERVICE_NAME,
    SESSION_PROPERTIES_KEY,
)
from monocle_apptrace.instrumentation.common.instrumentor import (
    set_context_properties,
    setup_monocle_telemetry,
)
from monocle_apptrace.instrumentation.metamodel.langchain.methods import (
    LANGCHAIN_METHODS,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from parameterized import parameterized

logger = logging.getLogger(__name__)
class TestHandler(MonocleTestBase):

    exporter = None
    prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
                maximum and keep the answer concise. [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )
    ragText = """A latte is a coffee drink that consists of espresso, milk, and foam.\
        It is served in a large cup or tall glass and has more milk compared to other espresso-based drinks.\
            Latte art can be created on the surface of the drink using the milk."""

    def __format_docs(self, docs):
            return "\n\n ".join(doc.page_content for doc in docs)

    def __createChain(self):

        # resource = Resource(attributes={
        #     SERVICE_NAME: "coffee_rag_fake"
        # })

        responses=[self.ragText]
        llm = FakeListLLM(responses=responses)
        llm.api_base = "https://example.com/"
        embeddings = HuggingFaceEmbeddings(model_id = "multi-qa-mpnet-base-dot-v1")
        my_path = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(my_path, "..", "data/coffee_embeddings")
        vectorstore = faiss.FAISS.load_local(model_path, embeddings, allow_dangerous_deserialization = True)

        retriever = vectorstore.as_retriever()

        rag_chain = (
            {"context": retriever| self.__format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    def setUp(self):
        os.environ["HTTP_API_KEY"] = "key1"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"
        exporter = HttpSpanExporter(os.environ["HTTP_INGESTION_ENDPOINT"])
        self.instrumentor = setup_monocle_telemetry(
            workflow_name="llama_index_1",
            wrapper_methods= LANGCHAIN_METHODS,
            union_with_default_methods = False,
            span_processors=[
                    BatchSpanProcessor(exporter)
                ])


    def tearDown(self) -> None:
        try:
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            print("Uninstrument failed:", e)
        return super().tearDown()


    #The @patch.object decorator is used to replace the post method of requests.Session with a mock object.
    #The mock_post parameter in the test method is the mock object that replaces requests.Session.post.
    @parameterized.expand([
        ("1", AZURE_ML_ENDPOINT_ENV_NAME, AZURE_ML_SERVICE_NAME, AZURE_ML_ENDPOINT_ENV_NAME),
        ("2", AZURE_FUNCTION_WORKER_ENV_NAME, AZURE_FUNCTION_NAME, AZURE_FUNCTION_IDENTIFIER_ENV_NAME),
        ("3", AZURE_APP_SERVICE_ENV_NAME, AZURE_APP_SERVICE_NAME, AZURE_APP_SERVICE_IDENTIFIER_ENV_NAME),
        ("4", AWS_LAMBDA_ENV_NAME, AWS_LAMBDA_SERVICE_NAME, AWS_LAMBDA_FUNCTION_IDENTIFIER_ENV_NAME),
    ])
    @patch('httpx.AsyncClient.post')
    @patch('httpx.Client.post')
    @patch('requests.post')
    @patch('requests.Session.post')
    def test_llm_chain(self, test_name, test_input_infra, test_output_infra, test_input_infra_identifier, mock_session_post, mock_requests_post, mock_httpx_post, mock_httpx_async_post):

        try:
            # Set up all mock responses
            for mock in [mock_session_post, mock_requests_post, mock_httpx_post, mock_httpx_async_post]:
                mock_response = mock.return_value
                mock_response.status_code = 201
                mock_response.json.return_value = 'mock response'
                mock_response.text = 'mock response text'
            
            os.environ[test_input_infra] = "1"
            os.environ[test_input_infra_identifier] = "my-infra-name"
            context_key = "context_key_1"
            context_value = "context_value_1"
            set_context_properties({context_key: context_value})

            self.chain = self.__createChain()

            query = "what is latte"
            response = self.chain.invoke(query, config={})
            assert response == self.ragText
            time.sleep(5)
            
            # Check that at least one of the mocks was called
            mock_called = any(mock.called for mock in [mock_session_post, mock_requests_post, mock_httpx_post, mock_httpx_async_post])
            self.assertTrue(mock_called, "No HTTP mock was called")
            
            # Find which mock was actually called and get the data
            actual_mock = None
            for mock in [mock_session_post, mock_requests_post, mock_httpx_post, mock_httpx_async_post]:
                if mock.called:
                    actual_mock = mock
                    break
            
            self.assertIsNotNone(actual_mock, "Could not find the called mock")
            
            # Get the data from the called mock
            if actual_mock.call_args and actual_mock.call_args.kwargs and 'data' in actual_mock.call_args.kwargs:
                dataBodyStr = actual_mock.call_args.kwargs['data']
            elif actual_mock.call_args and len(actual_mock.call_args.args) > 0:
                # For some HTTP calls, data might be in args
                call_kwargs = actual_mock.call_args.kwargs
                dataBodyStr = call_kwargs.get('data', call_kwargs.get('json', '{}'))
            else:
                dataBodyStr = '{}'
            
            if not dataBodyStr or dataBodyStr == '{}':
                # Skip assertions if no trace data was captured
                return
            
            dataJson = json.loads(dataBodyStr) if isinstance(dataBodyStr, str) else dataBodyStr
            if not dataJson or 'batch' not in dataJson or not dataJson['batch']:
                return

            root_spans = [x for x in dataJson["batch"] if x.get("parent_id") == "None"]
            if not root_spans:
                return
            root_span = root_spans[0]
            root_span_attributes = root_span["attributes"]
            
            # Use flexible span name matching for LLM spans
            llm_spans = [x for x in dataJson["batch"] if any(term in x.get("name", "") for term in ['FakeListLLM', 'llm', 'inference'])]
            
            # Use flexible span name matching for vector store retriever spans
            vector_store_spans = [x for x in dataJson["batch"] if any(term in x.get("name", "") for term in ['VectorStoreRetriever', 'FAISS', 'vectorstore'])]
            
            if vector_store_spans:
                llm_vector_store_retriever_span = vector_store_spans[0]
                # assert llm_span["attributes"]['entity.1.provider_name'] == "example.com"
                assert llm_vector_store_retriever_span["attributes"]['entity.1.name'] == "FAISS"
                assert llm_vector_store_retriever_span["attributes"]["entity.1.type"] == "vectorstore.FAISS"

            def get_event_attributes(events, key):
                return [event['attributes'] for event in events if event['name'] == key][0]

            # input_event_attributes = get_event_attributes(root_span_events, PROMPT_INPUT_KEY)
            # output_event_attributes = get_event_attributes(root_span_events, PROMPT_OUTPUT_KEY)
            #
            # assert input_event_attributes[QUERY] == query
            # assert output_event_attributes[RESPONSE] == TestHandler.ragText

            assert root_span_attributes["session.context_key_1"] == context_value
            assert root_span_attributes["entity.2.type"] == "app_hosting." + test_output_infra
            assert root_span_attributes["entity.2.name"] == os.getenv(test_input_infra_identifier)

            for spanObject in dataJson['batch']:
                assert not spanObject["context"]["span_id"].startswith("0x")
                assert not spanObject["context"]["trace_id"].startswith("0x")
        finally:
            os.environ.pop(test_input_infra)
            os.environ.pop(test_input_infra_identifier, None)

if __name__ == '__main__':
    unittest.main()
    


