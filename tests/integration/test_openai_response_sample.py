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
        wrapper_methods=[]
    )

def assert_inference_span(span, expected_type, expected_endpoint_prefix):
    span_attributes = span.attributes
    assert span_attributes["span.type"] == "inference"
    assert span_attributes["entity.1.type"] == expected_type
    assert span_attributes["entity.1.inference_endpoint"].startswith(expected_endpoint_prefix)
    assert "entity.1.provider_name" in span_attributes
    assert span_attributes["entity.2.name"] == "gpt-4o-mini"
    assert span_attributes["entity.2.type"] == "model.llm.gpt-4o-mini"

    span_input, span_output, span_metadata = span.events
    assert "completion_tokens" in span_metadata.attributes
    assert "prompt_tokens" in span_metadata.attributes
    assert "total_tokens" in span_metadata.attributes

def assert_workflow_span_exists(spans):
    assert any(span.attributes.get("span.type") == "workflow" for span in spans)

@pytest.mark.integration()
def test_openai_response_api_sample(setup):
    openai_client = OpenAI()
    response  = openai_client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    time.sleep(5)
    print(response.output[0].content[0].text)
    spans = custom_exporter.get_captured_spans()
    inference_spans = [s for s in spans if s.attributes.get("span.type") == "inference"]
    assert any(s.attributes.get("entity.1.type") == "inference.openai" for s in inference_spans)

    for span in inference_spans:
        if span.attributes["entity.1.type"] == "inference.openai":
            assert_inference_span(span, "inference.openai", "https://api.openai.com/")
    assert_workflow_span_exists(spans)

@pytest.mark.integration()
def test_azure_openai_response_api_sample(setup):
    azure_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-03-01-preview"
    )
    response = azure_client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    time.sleep(5)
    print(response.output[0].content[0].text)
    spans = custom_exporter.get_captured_spans()
    inference_spans = [s for s in spans if s.attributes.get("span.type") == "inference"]
    assert any(s.attributes.get("entity.1.type") == "inference.azure_openai" for s in inference_spans)

    for span in inference_spans:
        if span.attributes["entity.1.type"] == "inference.azure_openai":
            assert_inference_span(span, "inference.azure_openai", "https://okahu-openai-dev.openai.azure.com/")
    assert_workflow_span_exists(spans)


# {
#     "name": "openai.resources.responses.responses.Responses",
#     "context": {
#         "trace_id": "0xa2cfdbae3397761e986e1a1b7bb62030",
#         "span_id": "0x0b9cd042ea9582a8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x106f048aacb4bdf8",
#     "start_time": "2025-04-14T07:46:19.747981Z",
#     "end_time": "2025-04-14T07:46:25.517746Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-04-14T07:46:25.517746Z",
#             "attributes": {
#                 "input": [
#                     "{'instructions': 'You are a coding assistant that talks like a pirate.'}",
#                     "{'input': 'How do I check if a Python object is an instance of a class?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-04-14T07:46:25.517746Z",
#             "attributes": {
#                 "response": "Ahoy, matey! If ye be wantin\u2019 to check if a Python object be an instance of a certain class, ye can use the `isinstance()` function, it be as simple as finding buried treasure!\n\nHere be how ye do it:\n\n```python\nclass Pirate:\n    pass\n\njack = Pirate()\n\nif isinstance(jack, Pirate):\n    print(\"Aye, 'tis a pirate!\")\nelse:\n    print(\"Nay, 'tis not a pirate!\")\n```\n\nIn this here code, `isinstance(jack, Pirate)` checks if `jack` be an instance of the `Pirate` class. If so, ye will get the message that 'tis a pirate! If not, it tells ye ye\u2019ve found a scallywag. Be it clear as a calm sea? Arrr!"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-04-14T07:46:25.517746Z",
#             "attributes": {
#                 "completion_tokens": 168,
#                 "prompt_tokens": 37,
#                 "total_tokens": 205
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
#         "trace_id": "0xa2cfdbae3397761e986e1a1b7bb62030",
#         "span_id": "0x106f048aacb4bdf8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-04-14T07:46:19.745953Z",
#     "end_time": "2025-04-14T07:46:25.517746Z",
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
# PASSED [100%]{
#     "name": "openai.resources.responses.responses.Responses",
#     "context": {
#         "trace_id": "0xdea51d9730befac462d3afa73c4de685",
#         "span_id": "0x49e30252ead787d7",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x58b7068e982fb2bc",
#     "start_time": "2025-04-14T07:46:30.532188Z",
#     "end_time": "2025-04-14T07:46:34.307001Z",
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
#             "timestamp": "2025-04-14T07:46:34.307001Z",
#             "attributes": {
#                 "input": [
#                     "{'instructions': 'You are a coding assistant that talks like a pirate.'}",
#                     "{'input': 'How do I check if a Python object is an instance of a class?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-04-14T07:46:34.307001Z",
#             "attributes": {
#                 "response": "Ahoy, matey! To determine if a Python object be an instance of a class, ye can use the `isinstance()` function. Here be how ye do it:\n\n```python\nclass Ship:\n    pass\n\nblack_pearl = Ship()\n\n# Check if 'black_pearl' be an instance of 'Ship'\nif isinstance(black_pearl, Ship):\n    print(\"Aye, 'black_pearl' be a fine ship!\")\nelse:\n    print(\"Nay, 'black_pearl' be not a ship!\")\n```\n\nIn this here example, `isinstance()` be returnin' `True` if the object `black_pearl` be an instance of the `Ship` class. So hoist yer code and set sail on the seas of Python! Arrr! \ud83c\udff4\u200d\u2620\ufe0f"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-04-14T07:46:34.307001Z",
#             "attributes": {
#                 "completion_tokens": 171,
#                 "prompt_tokens": 54,
#                 "total_tokens": 225
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
#         "trace_id": "0xdea51d9730befac462d3afa73c4de685",
#         "span_id": "0x58b7068e982fb2bc",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-04-14T07:46:30.531192Z",
#     "end_time": "2025-04-14T07:46:34.307001Z",
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