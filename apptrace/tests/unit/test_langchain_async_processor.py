import json
import logging
import os
import time
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import ANY, MagicMock, patch

import requests
from common.embeddings_wrapper import HuggingFaceEmbeddings
from common.fake_list_llm import FakeListLLM
from common.http_span_exporter import HttpSpanExporter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import faiss
from langchain_core.runnables import RunnablePassthrough
from monocle_apptrace.instrumentation.common.instrumentor import (
    MonocleInstrumentor,
    set_context_properties,
    setup_monocle_telemetry,
)
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

events = []
logger = logging.getLogger(__name__)

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
        self.instrumentor = MonocleInstrumentor(handlers={"default":SpanHandler()})
        self.instrumentor.instrument()
        self.processor = monocleProcessor
        responses = [self.ragText]
        llm = FakeListLLM(responses=responses)
        llm.api_base = "https://example.com/"
        embeddings = HuggingFaceEmbeddings(model_id="multi-qa-mpnet-base-dot-v1")
        my_path = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(my_path, "..", "data/coffee_embeddings")
        vectorstore = faiss.FAISS.load_local(model_path, embeddings, allow_dangerous_deserialization=True)

        retriever = vectorstore.as_retriever()

        rag_chain = (
                {"context": retriever | self.__format_docs, "question": RunnablePassthrough()}
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
                    package="langchain.chat_models.base",
                    object_name="BaseChatModel",
                    method="invoke",
                    wrapper_method=atask_wrapper
                )

            ])
        try:
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
                url='https://localhost:3000/api/v1/traces',
                data=ANY,
                timeout=ANY
            )

            '''mock_post.call_args gives the parameters used to make post call.
            This can be used to do more asserts'''
            dataBodyStr = mock_post.call_args.kwargs['data']
            dataJson = json.loads(dataBodyStr)  # more asserts can be added on individual fields

            # Find the LLM generation span (the main span we want to verify)
            llm_generation_spans = [x for x in dataJson["batch"] if "langchain_core.language_models.llms.LLM._generate" in x["name"]]
            
            if not llm_generation_spans:
                # Debug: Print available span names if we can't find the expected one
                span_names = [x["name"] for x in dataJson["batch"]]
                raise AssertionError(f"No LLM._generate span found. Available spans: {span_names}")
            
            llm_span = llm_generation_spans[0]
            
            # Debug: Print the attributes to see what's available
            print(f"LLM span attributes: {llm_span.get('attributes', {})}")

            assert llm_span["attributes"]["span.type"] == "inference.framework"
            # assert llm_span["attributes"]["entity.1.provider_name"] == "example.com"
            assert llm_span["attributes"]["entity.1.type"] == "inference.generic"
            assert llm_span["attributes"]["entity.1.inference_endpoint"] == "https://example.com/"

        finally:
            try:
                if self.instrumentor is not None:
                    self.instrumentor.uninstrument()
            except Exception as e:
                logger.info("Uninstrument failed:", e)


if __name__ == "__main__":
    unittest.main()
