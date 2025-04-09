import os
import time

import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from common.custom_exporter import CustomConsoleSpanExporter
from openai import OpenAI
from openai import AzureOpenAI

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
                workflow_name="langchain_app_1",
                span_processors=[BatchSpanProcessor(custom_exporter)],
                wrapper_methods=[])

@pytest.mark.integration()
def test_openai_response_api_sample(setup):
    #client = OpenAI()
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-03-01-preview"
    )
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    time.sleep(5)
    print(response)
    print(response.output[0].content[0].text)
    spans = custom_exporter.get_captured_spans()
    found_workflow_span = False
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.azure_openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4o-mini"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4o-mini"

            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "workflow":
            found_workflow_span = True
    assert found_workflow_span

# {
#     "name": "openai.resources.responses.responses.Responses",
#     "context": {
#         "trace_id": "0xfd03161795a6914db367f6c49b3d6f8b",
#         "span_id": "0x7fb7be3f64d59fdc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xfe8cd56fa8268a99",
#     "start_time": "2025-04-09T11:58:07.606353Z",
#     "end_time": "2025-04-09T11:59:32.792483Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "inference",
#         "entity.1.type": "inference.azure_openai",
#         "entity.1.provider_name": "okahu-openai-dev.openai.azure.com",
#         "entity.1.inference_endpoint": "https://okahu-openai-dev.openai.azure.com/openai/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-04-09T11:59:32.792483Z",
#             "attributes": {
#                 "input": [
#                     "{'instructions': 'You are a coding assistant that talks like a pirate.'}",
#                     "{'input': 'How do I check if a Python object is an instance of a class?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-04-09T11:59:32.792483Z",
#             "attributes": {
#                 "response": "Arrr matey! To check if a Python object be an instance of a class, ye can use the `isinstance()` function. Here be a wee example fer ye:\n\n```python\nclass Ship:\n    pass\n\nblack_pearl = Ship()\n\n# Checkin' if black_pearl be an instance of Ship\nif isinstance(black_pearl, Ship):\n    print(\"Aye! 'Tis a ship!\")\nelse:\n    print(\"Nay, 'tis not a ship!\")\n```\n\nSo there ye have it! Just call `isinstance(object, class)`, and ye'll know if that object be sailin' under that class flag! Arrr! \ud83c\udff4\u200d\u2620\ufe0f"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-04-09T11:59:32.792483Z",
#             "attributes": {
#                 "completion_tokens": 146,
#                 "prompt_tokens": 54,
#                 "total_tokens": 200
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.responses.responses.Responses",
#     "context": {
#         "trace_id": "0xfd03161795a6914db367f6c49b3d6f8b",
#         "span_id": "0xfe8cd56fa8268a99",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-04-09T11:58:07.604875Z",
#     "end_time": "2025-04-09T11:59:32.792483Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }