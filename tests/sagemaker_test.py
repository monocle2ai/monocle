import json
import os
import time
import unittest
from unittest.mock import ANY, MagicMock, patch
import requests
from http_span_exporter import HttpSpanExporter
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from sagemaker_sample import produce_llm_response

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
            response = produce_llm_response(query)

            time.sleep(5)
            mock_post.assert_called_with(
                url='https://localhost:3000/api/v1/traces',
                data=ANY,
                timeout=ANY
            )

            dataBodyStr = mock_post.call_args.kwargs['data']
            dataJson = json.loads(dataBodyStr)
            boto_attributes = [x for x in dataJson["batch"] if x["parent_id"] == "None"][0]["attributes"]

            assert boto_attributes["span.type"] == "inference"

            assert boto_attributes["entity.2.type"] == "inference.aws_sagemaker"
            assert boto_attributes["entity.2.inference_endpoint"] == "https://runtime.sagemaker.us-east-1.amazonaws.com"

            assert boto_attributes["entity.3.name"] == "okahu-sagemaker-rag-qa-ep"
            assert boto_attributes["entity.3.type"] == "model.llm.okahu-sagemaker-rag-qa-ep"

            boto_events = [x for x in dataJson["batch"] if x["parent_id"] == "None"][0]["events"]

            assert boto_events[0]['attributes']['input'] == [query]

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


