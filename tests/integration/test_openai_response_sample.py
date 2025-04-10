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
    openai_client = OpenAI()
    openai_response = openai_client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    time.sleep(5)
    print(openai_response)
    print(openai_response.output[0].content[0].text)

    azure_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-03-01-preview"
    )
    azure_response = azure_client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    time.sleep(5)
    print(azure_response)
    print(azure_response.output[0].content[0].text)
    spans = custom_exporter.get_captured_spans()
    found_workflow_span = False
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            if span_attributes["entity.1.type"] == "inference.azure_openai":
                assert span_attributes["entity.1.type"] == "inference.azure_openai"
            else:
                assert span_attributes["entity.1.type"] == "inference.openai"
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
#         "trace_id": "0x430f1b448381a1cf14af821a36bef4fa",
#         "span_id": "0xd7ed6549fa4ed07f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa2a410702e509f62",
#     "start_time": "2025-04-10T05:17:10.961057Z",
#     "end_time": "2025-04-10T05:17:14.065572Z",
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
#             "timestamp": "2025-04-10T05:17:14.065572Z",
#             "attributes": {
#                 "input": [
#                     "{'instructions': 'You are a coding assistant that talks like a pirate.'}",
#                     "{'input': 'How do I check if a Python object is an instance of a class?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-04-10T05:17:14.065572Z",
#             "attributes": {
#                 "response": "Ahoy there, matey! To check if a Python object be an instance of a class, ye can use the `isinstance()` function. Here be the syntax:\n\n```python\nif isinstance(your_object, YourClass):\n    # yer code goes here\n```\n\nThis magic spell will return `True` if `your_object` be an instance of `YourClass` or any of its subclasses. If ye be wantin' to check against the class itself without regardin' subclasses, use the `type()` function like so:\n\n```python\nif type(your_object) is YourClass:\n    # yer code goes here\n```\n\nFair winds in yer coding seas, matey! \ud83c\udff4\u200d\u2620\ufe0f"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-04-10T05:17:14.065572Z",
#             "attributes": {
#                 "completion_tokens": 150,
#                 "prompt_tokens": 37,
#                 "total_tokens": 187
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
#         "trace_id": "0x430f1b448381a1cf14af821a36bef4fa",
#         "span_id": "0xa2a410702e509f62",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-04-10T05:17:10.960056Z",
#     "end_time": "2025-04-10T05:17:14.065572Z",
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
# {
#     "name": "openai.resources.responses.responses.Responses",
#     "context": {
#         "trace_id": "0xe19a3078a1afd7bb61e71a66864ae685",
#         "span_id": "0xe7c7ae5628c52b44",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xbf6da97876026f81",
#     "start_time": "2025-04-10T05:17:19.091743Z",
#     "end_time": "2025-04-10T05:17:22.482851Z",
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
#             "timestamp": "2025-04-10T05:17:22.482851Z",
#             "attributes": {
#                 "input": [
#                     "{'instructions': 'You are a coding assistant that talks like a pirate.'}",
#                     "{'input': 'How do I check if a Python object is an instance of a class?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-04-10T05:17:22.482851Z",
#             "attributes": {
#                 "response": "Ahoy there, matey! To check if a Python object be an instance o' a particular class, ye can use the `isinstance()` function. Here be how ye do it:\n\n```python\nclass Ship:\n    pass\n\nblack_pearl = Ship()\n\n# Check if black_pearl be an instance of the Ship class\nif isinstance(black_pearl, Ship):\n    print(\"Arrr! Black Pearl be a fine ship!\")\nelse:\n    print(\"Nay, that be not a ship!\")\n```\n\nIn this here example, `isinstance()` checks if `black_pearl` be an instance of the `Ship` class. If it be, ye get a hearty confirmation! If not, well, it be a sad tale indeed! \u2693\ufe0f"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-04-10T05:17:22.482851Z",
#             "attributes": {
#                 "completion_tokens": 159,
#                 "prompt_tokens": 54,
#                 "total_tokens": 213
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
#         "trace_id": "0xe19a3078a1afd7bb61e71a66864ae685",
#         "span_id": "0xbf6da97876026f81",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-04-10T05:17:19.089743Z",
#     "end_time": "2025-04-10T05:17:22.482851Z",
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