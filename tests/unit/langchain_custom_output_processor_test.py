
import json
import logging
import os
import time
import unittest
from unittest.mock import ANY, MagicMock, patch

import requests
from common.embeddings_wrapper import HuggingFaceEmbeddings
from common.fake_list_llm import FakeListLLM
from common.http_span_exporter import HttpSpanExporter
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import faiss
from langchain_core.runnables import RunnablePassthrough
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from parameterized import parameterized

from monocle_apptrace.instrumentation.common.constants import AZURE_ML_ENDPOINT_ENV_NAME
from monocle_apptrace.instrumentation.common.instrumentor import (
    MonocleInstrumentor,
    set_context_properties,
    setup_monocle_telemetry,
)
from monocle_apptrace.instrumentation.common.span_handler import (
    WORKFLOW_TYPE_MAP,
    SpanHandler,
)

logger = logging.getLogger(__name__)

class TestHandler(unittest.TestCase):

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

    def __createChain(self, llm_type):

        resource = Resource(attributes={
            SERVICE_NAME: "coffee_rag_fake"
        })
        traceProvider = TracerProvider(resource=resource)
        exporter = ConsoleSpanExporter()
        monocleProcessor = BatchSpanProcessor(exporter)

        traceProvider.add_span_processor(monocleProcessor)
        trace.set_tracer_provider(traceProvider)
        self.instrumentor = MonocleInstrumentor(handlers=SpanHandler())
        self.instrumentor.instrument()
        self.processor = monocleProcessor
        responses = [self.ragText]
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


    def tearDown(self) -> None:
        return super().tearDown()

    @parameterized.expand([
        ("FakeListLLM", AZURE_ML_ENDPOINT_ENV_NAME, "FakeListLLM"),
    ])
    @patch.object(requests.Session, 'post')
    def test_llm_chain(self, test_name, test_input_infra, llm_type, mock_post):
        app_name = "test"
        wrap_method = MagicMock(return_value=3)
        setup_monocle_telemetry(
            workflow_name=app_name,
            span_processors=[
                BatchSpanProcessor(HttpSpanExporter("https://localhost:3000/api/v1/traces"))
            ],
            wrapper_methods=[
            ])
        try:

            os.environ[test_input_infra] = "1"
            context_key = "context_key_1"
            context_value = "context_value_1"
            set_context_properties({context_key: context_value})

            self.chain = self.__createChain(llm_type)
            mock_post.return_value.status_code = 201
            mock_post.return_value.json.return_value = 'mock response'

            query = "what is latte"
            response = self.chain.invoke(query, config={})
            # assert response == self.ragText
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
            root_attributes = [x for x in dataJson["batch"] if x['attributes'].get("span.type") == 'workflow'][0]["attributes"]
            assert root_attributes["entity.1.name"] == app_name
            assert root_attributes["entity.1.type"] == WORKFLOW_TYPE_MAP['langchain']
            if llm_type == "FakeListLLM":
                llm_vector_store_retriever_span = [x for x in dataJson["batch"] if 'langchain_core.vectorstores.base.VectorStoreRetriever' in x["name"]][0]
                inference_span = [x for x in dataJson["batch"] if 'fake_list_llm.FakeListLLM' in x["name"]][0]

                assert llm_vector_store_retriever_span["attributes"]["span.type"] == "retrieval"
                assert llm_vector_store_retriever_span["attributes"]["entity.1.name"] == "FAISS"
                assert llm_vector_store_retriever_span["attributes"]["entity.1.type"] == "vectorstore.FAISS"
                assert inference_span['attributes']["entity.1.inference_endpoint"] == "https://example.com/"

        finally:
            os.environ.pop(test_input_infra)
            try:
                if(self.instrumentor is not None):
                    self.instrumentor.uninstrument()
            except Exception as e:
                print("Uninstrument failed:", e)


if __name__ == '__main__':
    unittest.main()


