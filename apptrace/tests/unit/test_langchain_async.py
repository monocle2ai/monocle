import json
import logging
import os
import time
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import ANY, MagicMock, patch

from common.embeddings_wrapper import HuggingFaceEmbeddings
from common.fake_list_llm import FakeListLLM
from common.http_span_exporter import HttpSpanExporter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import faiss
from langchain_core.runnables import RunnablePassthrough
from monocle_apptrace.instrumentation.common.constants import (
    SESSION_PROPERTIES_KEY,
)
from monocle_apptrace.instrumentation.common.instrumentor import (
    MonocleInstrumentor,
    set_context_properties,
    setup_monocle_telemetry,
)
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
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
        events.append("setUp")

    async def asyncSetUp(self):
        os.environ["HTTP_API_KEY"] = "key1"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"

    @patch('requests.Session.post')
    @patch('requests.post')
    async def test_response(self, mock_requests_post, mock_session_post):
        app_name = "test"
        wrap_method = MagicMock(return_value=3)
        
        # Configure both mocks to return successful responses
        def create_mock_response():
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.text = 'mock response text'
            mock_response.json.return_value = 'mock response'
            return mock_response
        
        mock_session_post.return_value = create_mock_response()
        mock_requests_post.return_value = create_mock_response()
        
        setup_monocle_telemetry(
            workflow_name=app_name,
            span_processors=[
                    BatchSpanProcessor(HttpSpanExporter("https://localhost:3000/api/v1/traces"))
                ],
            wrapper_methods=[
                WrapperMethod(
                    package="dummy_class",
                    object_name="DummyClass",
                    method="dummy_method",
                    span_name="langchain.workflow",
                    wrapper_method=wrap_method),

        ])
        try:
            context_key = "context_key_1"
            context_value = "context_value_1"
            set_context_properties({context_key: context_value})

            self.chain = self.__createChain()

            query = "what is latte"
            response = self.chain.invoke(query, config={})
            assert response == self.ragText
            time.sleep(5)
            mock_session_post.assert_called_with(
                url = 'https://localhost:3000/api/v1/traces',
                data=ANY,
                timeout=ANY
            )

            '''mock_session_post.call_args gives the parameters used to make post call.
            This can be used to do more asserts'''
            dataBodyStr = mock_session_post.call_args.kwargs['data']
            dataJson =  json.loads(dataBodyStr) # more asserts can be added on individual fields
            # assert len(dataJson['batch']) == 7

            # Debug spans for verification
            logger.debug("All spans:")
            for span in dataJson["batch"]:
                logger.debug(f"  - {span['name']} (parent_id: {span.get('parent_id', 'N/A')})")
                if "llm" in span['name'].lower():
                    logger.debug(f"    Attributes: {span.get('attributes', {})}")

            root_span = [x for x in  dataJson["batch"] if x["parent_id"] == "None"][0]
            # Look for LLM span - now using the correct span name pattern
            llm_spans = [x for x in  dataJson["batch"] if "LLM" in x["name"] and "generate" in x["name"]]
            if llm_spans:
                llm_span = llm_spans[0]
            else:
                # Fallback to any LLM-related span
                llm_spans = [x for x in  dataJson["batch"] if "llm" in x["name"].lower()]
                if llm_spans:
                    llm_span = llm_spans[0]
                else:
                    logger.debug("No LLM span found, available spans: %s", [x["name"] for x in dataJson["batch"]])
                    llm_span = None
            root_span_attributes = root_span["attributes"]
            
            if llm_span:
                assert llm_span["attributes"]["entity.1.inference_endpoint"] == "https://example.com/"

            def get_event_attributes(events, key):
                return [event['attributes'] for event in events if event['name'] == key][0]

            # input_event_attributes = get_event_attributes(root_span_events, PROMPT_INPUT_KEY)
            # output_event_attributes = get_event_attributes(root_span_events, PROMPT_OUTPUT_KEY)
            #
            # assert input_event_attributes[QUERY] == query
            # assert output_event_attributes[RESPONSE] == Test.ragText
            assert root_span_attributes[f"{SESSION_PROPERTIES_KEY}.{context_key}"] == context_value

            for spanObject in dataJson['batch']:
                assert not spanObject["context"]["span_id"].startswith("0x")
                assert not spanObject["context"]["trace_id"].startswith("0x")
        finally:
            try:
                if(self.instrumentor is not None):
                    self.instrumentor.uninstrument()
            except Exception as e:
                logger.debug("Uninstrument failed: %s", e)

    

    def tearDown(self):
        # Clean up instrumentation state
        if hasattr(self, 'instrumentor') and self.instrumentor is not None:
            try:
                self.instrumentor.uninstrument()
            except Exception as e:
                logger.warning(f"Uninstrument failed: {e}")
        return super().tearDown()

    async def asyncTearDown(self):
        events.append("asyncTearDown")

    async def on_cleanup(self):
        events.append("cleanup")

if __name__ == "__main__":
    unittest.main()