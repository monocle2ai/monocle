import json
import logging
import os
import time
import unittest
from typing import List
from unittest.mock import ANY, patch
from haystack.components.builders.prompt_builder import PromptBuilder
import requests
from haystack import Pipeline, Document
from http_span_exporter import HttpSpanExporter
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.span_handler import WORKFLOW_TYPE_MAP
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

logger = logging.getLogger(__name__)


class TestHandler(unittest.TestCase):
    ragText = "sample_rag_text"
    os.environ["HTTP_API_KEY"] = "key1"
    os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"
    @patch.object(requests.Session, 'post')
    def test_haystack(self, mock_post):
        api_key = os.getenv("OPENAI_API_KEY")
        workflow_name = "haystack_app_1"
        setup_monocle_telemetry(
            workflow_name=workflow_name,
            span_processors=[BatchSpanProcessor(HttpSpanExporter("https://localhost:3000/api/v1/traces"))],
            wrapper_methods=[

            ]
        )
        documents = [Document(content="Joe lives in Berlin"), Document(content="Joe is a software engineer")]

        prompt_template = """
            Given these documents, answer the question.\nDocuments:
            {% for doc in documents %}
                {{ doc.content }}
            {% endfor %}

            \nQuestion: {{query}}
            \nAnswer:
            """
        prompt_builder = PromptBuilder(template=prompt_template)
        if api_key is None:
            raise ValueError("API key must not be None")
        llm = OpenAIGenerator(api_key=Secret.from_token(api_key), model="gpt-3.5-turbo-0125")
        # llm = OpenAIChatGenerator(api_key=Secret.from_token(api_key), model="gpt-3.5-turbo")
        document_store = InMemoryDocumentStore()
        for doc in documents:
            document_store.write_documents([doc])
        retriever = InMemoryBM25Retriever(document_store=document_store)

        pipe = Pipeline()
        pipe.add_component("retriever", retriever)
        print('reteriver in pipe is', pipe)
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.add_component("llm", llm)
        pipe.connect("retriever", "prompt_builder.documents")
        pipe.connect("prompt_builder.prompt", "llm.prompt")
        query = "OpenTelemetry"
        message = f"Tell me a joke about {query}"
        pipe.run(
            {
                "retriever": {"query": message},
                "prompt_builder": {"query": message},
            }
        )

        time.sleep(3)
        '''mock_post.call_args gives the parameters used to make post call.
        This can be used to do more asserts'''
        dataBodyStr = mock_post.call_args.kwargs['data']
        # logger.debug(dataBodyStr)
        dataJson = json.loads(dataBodyStr)  # more asserts can be added on individual fields

        root_attributes = [x for x in dataJson["batch"] if x['parent_id'] == 'None'][0]["attributes"]
        # assert root_attributes["workflow_input"] == query
        # assert root_attributes["workflow_output"] == llm.dummy_response
        assert root_attributes["entity.1.name"] == workflow_name
        assert root_attributes["entity.1.type"] == WORKFLOW_TYPE_MAP["haystack"]

        assert len(dataJson['batch']) == 2
        # llmspan = dataJson["batch"].find

        assert dataJson["batch"][0]["attributes"]["span.type"] == "inference"
        span_names: List[str] = [span["name"] for span in dataJson['batch']]
        for name in ["haystack.components.generators.openai.OpenAIGenerator",
                     "haystack.core.pipeline.pipeline.Pipeline"]:
            assert name in span_names

        type_found = False
        model_name_found = False
        provider_found = False
        input_event = False

        for span in dataJson["batch"]:
            if span["name"] == "haystack.components.generators.openai.OpenAIGenerator":
                assert span["attributes"]["entity.count"] == 2
                assert span["attributes"]["entity.1.type"] == "inference.azure_oai"
                # assert span["attributes"]["entity.1.provider_name"] == "api.openai.com"
                assert span["attributes"]["entity.2.name"] == "gpt-3.5-turbo-0125"
                type_found = True
                provider_found = True
                model_name_found = True
                for event in span['events']:
                    if event['name'] == "data.input":
                        assert event['attributes']['input'] == [str({'user': message})]
                        input_event = True


        assert type_found
        assert model_name_found
        assert provider_found
        assert input_event


if __name__ == '__main__':
    unittest.main()
