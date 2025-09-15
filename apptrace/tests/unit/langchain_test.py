

import json
import logging
import os
import time
import unittest
from unittest.mock import ANY, patch

import requests
from common.embeddings_wrapper import HuggingFaceEmbeddings
from common.fake_list_llm import FakeListLLM
from common.http_span_exporter import HttpSpanExporter
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import faiss
from langchain_core.runnables import RunnablePassthrough
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from parameterized import parameterized

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

logger = logging.getLogger(__name__)
class TestHandler(unittest.TestCase):

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
    @patch.object(requests.Session, 'post') 
    def test_llm_chain(self, test_name, test_input_infra, test_output_infra, test_input_infra_identifier, mock_post):

        try:
            os.environ[test_input_infra] = "1"
            os.environ[test_input_infra_identifier] = "my-infra-name"
            context_key = "context_key_1"
            context_value = "context_value_1"
            set_context_properties({context_key: context_value})

            self.chain = self.__createChain()
            mock_post.return_value.status_code = 201
            mock_post.return_value.json.return_value = 'mock response'

            query = "what is latte"
            response = self.chain.invoke(query, config={})
            assert response == self.ragText
            time.sleep(5)
            mock_post.assert_called_with(
                url = 'https://localhost:3000/api/v1/traces',
                data=ANY,
                timeout=ANY
            )

            '''mock_post.call_args gives the parameters used to make post call.
            This can be used to do more asserts'''
            dataBodyStr = mock_post.call_args.kwargs['data']
            dataJson =  json.loads(dataBodyStr) # more asserts can be added on individual fields
            # assert len(dataJson['batch']) == 75 = {dict: 11} {'attributes': {'session.context_key_1': 'context_value_1'}, 'context': {'span_id': '18a8d75ec4c94523', 'trace_id': '4fedeffc8d9a4ec8b3029a437b667e15', 'trace_state': '[]'}, 'end_time': '2024-09-17T07:30:36.830264Z', 'events': [], 'kind': 'SpanKind.INTERNAL', 'links': [], 'name': 'langchain.task.StrOutputParser', 'parent_id': 'f371def04dfa963d', 'resource': {'attributes': {'service.name': 'test'}, 'schema_url': ''}, 'start_time': '2024-09-17T07:30:36.829211Z', 'status': {'status_code': 'UNSET'}}... View

            root_span = [x for x in dataJson["batch"] if x["parent_id"] == "None"][0]
            llm_span = [x for x in dataJson["batch"] if "FakeListLLM" in x["name"]][0]
            llm_vector_store_retriever_span = [x for x in dataJson["batch"] if 'langchain_core.vectorstores.base.VectorStoreRetriever' in x["name"]][0]
            root_span_attributes = root_span["attributes"]
            root_span_events = root_span["events"]
            
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
    


