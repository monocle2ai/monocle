

import json
import logging
import os
import os.path
import time
import unittest
from unittest.mock import ANY, patch

import requests
import torch
from common.http_span_exporter import HttpSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from transformers import GPT2DoubleHeadsModel, GPT2Tokenizer

logger = logging.getLogger(__name__)

class TestHandler(unittest.TestCase):
    instrumentor = None

    def setUp(self):
        """Set up test environment with clean state"""
        os.environ["HTTP_API_KEY"] = "key1"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"

    def tearDown(self):
        """Clean up instrumentation state"""
        if self.instrumentor is not None:
            try:
                self.instrumentor.uninstrument()
            except Exception as e:
                logger.warning(f"Uninstrument failed: {e}")

    @patch.object(requests.Session, 'post')
    def test_pytorch(self, mock_post):
        os.environ["HTTP_API_KEY"] = "key1"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"

        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = 'mock response'

        self.instrumentor = setup_monocle_telemetry(
            workflow_name="pytorch_1",
            span_processors=[BatchSpanProcessor(HttpSpanExporter(os.environ["HTTP_INGESTION_ENDPOINT"]))],
            wrapper_methods=[
                        WrapperMethod(
                            package="transformers",
                            object_name="GPT2DoubleHeadsModel",
                            method="forward",
                            span_name="pytorch.transformer.GPT2DoubleHeadsModel",
                            output_processor="output_processor",
                            wrapper_method=task_wrapper),
                    ],
            union_with_default_methods= False
            )
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

            # Add a [CLS] to the vocabulary (we should train it also!)
            num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

            embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

            choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            encoded_choices = [tokenizer.encode(s) for s in choices]
            cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

            input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
            mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1
            # the trace gets generated for the forward method which gets called here
            outputs = model(input_ids, mc_token_ids=mc_token_ids)
            lm_prediction_scores, mc_prediction_scores = outputs[:2]

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
            
            # The test should expect 2 spans: pytorch span + workflow span
            assert len(dataJson['batch']) == 2
            
            # Look for the pytorch span specifically
            pytorch_spans = [span for span in dataJson['batch'] if "pytorch.transformer.GPT2DoubleHeadsModel" in span.get("name", "")]
            assert len(pytorch_spans) == 1, f"Expected exactly 1 pytorch span, but found: {[s.get('name') for s in dataJson['batch']]}"
            
            # Assert on the pytorch span
            pytorch_span = pytorch_spans[0]
            assert pytorch_span["name"] == "pytorch.transformer.GPT2DoubleHeadsModel"
            
            # Verify workflow span is also present
            workflow_spans = [span for span in dataJson['batch'] if span.get("name") == "workflow"]
            assert len(workflow_spans) == 1, "Expected exactly 1 workflow span"
        
        finally:
            pass  # cleanup handled in tearDown

if __name__ == '__main__':
    unittest.main()
