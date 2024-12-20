
import json
import logging
import os
import time
import unittest
from typing import List
from unittest.mock import ANY, patch

import requests
from haystack import Pipeline
from haystack.components.builders import DynamicChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

logger = logging.getLogger(__name__)


class TestHandler(unittest.TestCase):
    ragText = "sample_rag_text"
    @patch.object(requests.Session, 'post')
    def test_haystack(self, mock_post):
        api_key = os.getenv("OPENAI_API_KEY")

        setup_monocle_telemetry(
            workflow_name="haystack_app_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[

                    ]
            )
        prompt_builder = DynamicChatPromptBuilder()
        llm = OpenAIChatGenerator(api_key=Secret.from_token(api_key), model="gpt-4")
        document_store = InMemoryDocumentStore()
        retriever = InMemoryBM25Retriever(document_store=document_store)

        pipe = Pipeline()
        pipe.add_component("retriever", retriever)
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.add_component("llm", llm)
        pipe.connect("retriever", "prompt_builder.template_variables")
        pipe.connect("prompt_builder.prompt", "llm.messages")
        query = "OpenTelemetry"
        messages = [ChatMessage.from_user("Tell me a joke about {{query}}")]
        pipe.run(
            data={
                "prompt_builder": {
                    "template_variables": {"query": query},
                    "prompt_source": messages,
                }
            }
        )

        time.sleep(3)
        mock_post.assert_called_with(
            url = 'https://localhost:3000/api/v1/traces',
            data=ANY,
            timeout=ANY
        )
        '''mock_post.call_args gives the parameters used to make post call.
        This can be used to do more asserts'''
        dataBodyStr = mock_post.call_args.kwargs['data']
        logger.debug(dataBodyStr)
        dataJson = json.loads(dataBodyStr) # more asserts can be added on individual fields

        root_attributes = [x for x in dataJson["batch"] if x["parent_id"] == "None"][0]["attributes"]
        # assert root_attributes["workflow_input"] == query
        # assert root_attributes["workflow_output"] == llm.dummy_response

        assert len(dataJson['batch']) == 2
        # llmspan = dataJson["batch"].find

        # assert dataJson["batch"][1]["attributes"]["workflow_type"] == "workflow.llamaindex"
        span_names: List[str] = [span["name"] for span in dataJson['batch']]
        for name in ["haystack.openai", "haystack_pipeline.workflow"]:
            assert name in span_names

        type_found = False
        model_name_found = False
        provider_found = False
        assert root_attributes["workflow_input"] == query
        assert root_attributes["workflow_output"] == TestHandler.ragText

        for span in dataJson["batch"]:
            if span["name"] == "haystack_pipeline.workflow" and "workflow_type" in span["attributes"]:
                assert span["attributes"]["workflow_type"] == "workflow.haystack"
                type_found = True
            if span["name"] == "haystack.openai" and "model_name" in span["attributes"]:
                assert span["attributes"]["model_name"] == "gpt-4"
                model_name_found = True
            if span["name"] == "haystack.retriever" and "type" in span["attributes"]:
                assert span["attributes"]["provider_name"] == "InMemoryDocumentStore"
                provider_found = True

        assert type_found
        assert model_name_found
        assert provider_found

if __name__ == '__main__':
    unittest.main()
