import json
import os
import time
import unittest
from unittest.mock import ANY, MagicMock, patch

import requests
from torch.fx.experimental.unification.unification_tools import assoc

from http_span_exporter import HttpSpanExporter
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from bedrock_opensearch_sample import produce_response

class TestSagemaker(unittest.TestCase):

    def setUp(self):
        os.environ["HTTP_API_KEY"] = "key1"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"

    @patch.object(requests.Session, 'post')
    def test_sagemaker_workflow(self, mock_post):
        app_name = "test"
        wrap_method = MagicMock(return_value=3)
        setup_monocle_telemetry(
            workflow_name=app_name,
            span_processors=[
                BatchSpanProcessor(HttpSpanExporter("https://localhost:3000/api/v1/traces"))
            ],
            wrapper_methods=[]
        )

        try:
            query = "what is latte"
            response = produce_response(query)

            time.sleep(5)
            mock_post.assert_called_with(
                url='https://localhost:3000/api/v1/traces',
                data=ANY,
                timeout=ANY
            )

            dataBodyStr = mock_post.call_args.kwargs['data']
            dataJson = json.loads(dataBodyStr)
            opensearch_attribute = [x for x in dataJson["batch"] if x["parent_id"] == "None"][0]["attributes"]

            assert opensearch_attribute["span.type"] == "retrieval"
            assert opensearch_attribute["entity.2.name"] == "OpenSearchVectorSearch"
            assert opensearch_attribute["entity.2.deployment"] =="https://vvd9mtj8odrs1h09sul4.us-east-1.aoss.amazonaws.com:443"
            assert opensearch_attribute["entity.2.type"] == "vectorstore.OpenSearchVectorSearch"
            assert opensearch_attribute["entity.3.name"] ==  "amazon.titan-embed-text-v1"
            assert opensearch_attribute["entity.3.type"] == "model.embedding.amazon.titan-embed-text-v1"

            boto_attributes = [x for x in dataJson["batch"] if x["parent_id"] == "None"][1]["attributes"]

            assert boto_attributes["span.type"] == "inference"

            assert boto_attributes["entity.2.type"] == "inference.aws_sagemaker"
            assert boto_attributes["entity.2.inference_endpoint"] == "https://bedrock-runtime.us-east-1.amazonaws.com"

            assert boto_attributes["entity.3.name"] == "ai21.j2-mid-v1"
            assert boto_attributes["entity.3.type"] == "model.llm.ai21.j2-mid-v1"

            boto_events = [x for x in dataJson["batch"] if x["parent_id"] == "None"][1]["events"]
            user_query = boto_events[0]['attributes']['input'][0]
            user_question = eval(user_query)['user']
            assert user_question == query

        finally:
            if "HTTP_API_KEY" in os.environ:
                os.environ.pop("HTTP_API_KEY")
            if "HTTP_INGESTION_ENDPOINT" in os.environ:
                os.environ.pop("HTTP_INGESTION_ENDPOINT")
            try:
                if hasattr(self, "instrumentor") and self.instrumentor is not None:
                    self.instrumentor.uninstrument()
            except Exception as e:
                print("Uninstrument failed:", e)


if __name__ == '__main__':
    unittest.main()

