
import json
import logging
import os
import time

import unittest
from unittest.mock import ANY, MagicMock, patch

import pytest
import requests
from dummy_class import DummyClass
from embeddings_wrapper import HuggingFaceEmbeddings
from http_span_exporter import HttpSpanExporter
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import faiss
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables import RunnablePassthrough
from monocle_apptrace.constants import (
    AZURE_APP_SERVICE_ENV_NAME,
    AZURE_APP_SERVICE_NAME,
    AZURE_FUNCTION_NAME,
    AZURE_FUNCTION_WORKER_ENV_NAME,
    AZURE_ML_ENDPOINT_ENV_NAME,
    AZURE_ML_SERVICE_NAME,
    AWS_LAMBDA_ENV_NAME,
    AWS_LAMBDA_SERVICE_NAME
)
from monocle_apptrace.instrumentor import (
    MonocleInstrumentor,
    set_context_properties,
    setup_monocle_telemetry,
)
from monocle_apptrace.wrap_common import (
    SESSION_PROPERTIES_KEY,
    INFRA_SERVICE_KEY,
    PROMPT_INPUT_KEY,
    PROMPT_OUTPUT_KEY,
    QUERY,
    RESPONSE,
    update_span_from_llm_response,
)
from monocle_apptrace.wrapper import WrapperMethod
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from fake_list_llm import FakeListLLM
from parameterized import parameterized

from monocle_apptrace.wrap_common import task_wrapper

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler('traces.txt' ,'w')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

class TestWorkflowEntityProperties(unittest.TestCase):

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

        resource = Resource(attributes={
            SERVICE_NAME: "coffee_rag_fake"
        })
        traceProvider = TracerProvider(resource=resource)
        exporter = ConsoleSpanExporter()
        monocleProcessor = BatchSpanProcessor(exporter)

        traceProvider.add_span_processor(monocleProcessor)
        trace.set_tracer_provider(traceProvider)
        self.instrumentor = MonocleInstrumentor()
        self.instrumentor.instrument()
        self.processor = monocleProcessor
        responses =[self.ragText]
        llm = FakeListLLM(responses=responses)
        llm.api_base = "https://example.com/"
        embeddings = HuggingFaceEmbeddings(model_id = "multi-qa-mpnet-base-dot-v1")
        my_path = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(my_path, "./vector_data/coffee_embeddings")
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


    def tearDown(self) -> None:
        return super().tearDown()

    @parameterized.expand([
        ("1", AZURE_ML_ENDPOINT_ENV_NAME, AZURE_ML_SERVICE_NAME),
        ("2", AZURE_FUNCTION_WORKER_ENV_NAME, AZURE_FUNCTION_NAME),
        ("3", AZURE_APP_SERVICE_ENV_NAME, AZURE_APP_SERVICE_NAME),
        ("4", AWS_LAMBDA_ENV_NAME, AWS_LAMBDA_SERVICE_NAME),
    ])
    @patch.object(requests.Session, 'post')
    def test_llm_chain(self, test_name, test_input_infra, test_output_infra, mock_post):
        app_name = "test"
        wrap_method = MagicMock(return_value=3)
        setup_monocle_telemetry(
            workflow_name=app_name,
            span_processors=[
                BatchSpanProcessor(HttpSpanExporter("https://localhost:3000/api/v1/traces"))
            ],
            wrapper_methods=[])
        try:

            os.environ[test_input_infra] = "1"
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

            root_span = [x for x in dataJson["batch"] if x["parent_id"] == "None"][0]

            # workflow_name and workflow_type in new format entity.{index}.name and entity.{index}.type

            assert root_span["attributes"]["entity.1.name"] == "test"
            assert root_span["attributes"]["entity.1.type"] == "workflow.langchain"

        finally:
            os.environ.pop(test_input_infra)
            try:
                if(self.instrumentor is not None):
                    self.instrumentor.uninstrument()
            except Exception as e:
                print("Uninstrument failed:", e)


if __name__ == '__main__':
    unittest.main()


