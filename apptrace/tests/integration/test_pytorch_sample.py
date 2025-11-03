

import logging

import pytest
import torch
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from transformers import GPT2DoubleHeadsModel, GPT2Tokenizer

logger = logging.getLogger(__name__)
@pytest.fixture(scope="module")
def setup():
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="pytorch_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[
                        WrapperMethod(
                            package="transformers",
                            object_name="GPT2DoubleHeadsModel",
                            method="forward",
                            span_name="pytorch.transformer.GPT2DoubleHeadsModel",
                            output_processor="output_processor",
                            wrapper_method=task_wrapper),
                        WrapperMethod(
                            package="transformers",
                            object_name="PreTrainedModel",
                            method="from_pretrained",
                            span_name="pytorch.transformer.PreTrainedModel",
                            output_processor="output_processor",
                            wrapper_method=task_wrapper),
                    ]
            )
        yield
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def test_pytorch_sample(setup):

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
    logger.info("done")

#{
#     "name": "pytorch.transformer.PreTrainedModel",
#     "context": {
#         "trace_id": "0x366ea97bb29057998c76a193b14f415a",
#         "span_id": "0x0eabc5f3834b163f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": 'None',
#     "start_time": "2024-04-16T15:43:54.793812Z",
#     "end_time": "2024-04-16T15:43:56.084558Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "workflow_name": "pytorch_1"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "pytorch_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "pytorch.transformer.GPT2DoubleHeadsModel",
#     "context": {
#         "trace_id": "0x99cf0b830f24095302fcd8443b4ec6b1",
#         "span_id": "0x9737b983237ec8dc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": 'None',
#     "start_time": "2024-04-16T15:44:03.978172Z",
#     "end_time": "2024-04-16T15:44:04.415499Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "workflow_name": "pytorch_1"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "pytorch_1"
#         },
#         "schema_url": ""
#     }
# }

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
