from unittest import IsolatedAsyncioTestCase
import unittest


import json
import logging
import os
import time

import unittest
from unittest.mock import ANY, MagicMock, patch
from unittest import IsolatedAsyncioTestCase

import requests
from dummy_class import DummyClass
from embeddings_wrapper import HuggingFaceEmbeddings
from http_span_exporter import HttpSpanExporter
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import faiss
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables import RunnablePassthrough
from monocle_apptrace.instrumentor import (
    MonocleInstrumentor,
    set_context_properties,
    setup_monocle_telemetry,
)
from monocle_apptrace.wrap_common import (
    CONTEXT_PROPERTIES_KEY,
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

from FakeListLLM import FakeListLLM

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler('traces.txt','w')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
events = []


class Test(IsolatedAsyncioTestCase):

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
        responses=[self.ragText]
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
        events.append("setUp")

    async def asyncSetUp(self):
        os.environ["HTTP_API_KEY"] = "key1"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"

    @patch.object(requests.Session, 'post')
    async def test_response(self, mock_post):
        app_name = "test"
        wrap_method = MagicMock(return_value=3)
        setup_monocle_telemetry(
            workflow_name=app_name,
            span_processors=[
                    BatchSpanProcessor(HttpSpanExporter("https://localhost:3000/api/v1/traces"))
                ],
            wrapper_methods=[
                WrapperMethod(
                    package="dummy_class",
                    object="DummyClass",
                    method="dummy_method",
                    span_name="langchain.workflow",
                    wrapper=wrap_method()),

        ])
        try:
            context_key = "context_key_1"
            context_value = "context_value_1"
            set_context_properties({context_key: context_value})

            self.chain = self.__createChain()
            mock_post.return_value.status_code = 201
            mock_post.return_value.json.return_value = 'mock response'

            query = "what is latte"
            response = await self.chain.ainvoke(query, config={})
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
            # assert len(dataJson['batch']) == 7

            root_span = [x for x in  dataJson["batch"] if x["parent_id"] == "None"][0]
            llm_span = [x for x in  dataJson["batch"] if "FakeListLLM" in x["name"]][0]
            root_span_attributes = root_span["attributes"]
            root_span_events = root_span["events"]
            
            assert llm_span["attributes"]["provider_name"] == "example.com"

            def get_event_attributes(events, key):
                return [event['attributes'] for event in events if event['name'] == key][0]

            input_event_attributes = get_event_attributes(root_span_events, PROMPT_INPUT_KEY)
            output_event_attributes = get_event_attributes(root_span_events, PROMPT_OUTPUT_KEY)
            
            assert input_event_attributes[QUERY] == query
            assert output_event_attributes[RESPONSE] == Test.ragText
            assert root_span_attributes[f"{CONTEXT_PROPERTIES_KEY}.{context_key}"] == context_value

            for spanObject in dataJson['batch']:
                assert not spanObject["context"]["span_id"].startswith("0x")
                assert not spanObject["context"]["trace_id"].startswith("0x")
        finally:
            try:
                if(self.instrumentor is not None):
                    self.instrumentor.uninstrument()
            except Exception as e:
                print("Uninstrument failed:", e)

    

    def tearDown(self):
        return super().tearDown()

    async def asyncTearDown(self):
        events.append("asyncTearDown")

    async def on_cleanup(self):
        events.append("cleanup")

if __name__ == "__main__":
    unittest.main()