# Enable Monocle Tracing
import time

from monocle_apptrace.instrumentation.common.instrumentor import (
    set_context_properties,
    setup_monocle_telemetry,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
import os
from google import genai
from google.genai import types

import pytest
custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
                workflow_name="gemini_app_1",
                span_processors=[BatchSpanProcessor(custom_exporter)],
                wrapper_methods=[])

@pytest.mark.integration()
def test_gemini_model_sample(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a cat. Your name is Neko."),
            contents="Hello there"
    )
    time.sleep(5)
    print(response.text)
    spans = custom_exporter.get_captured_spans()
    check_span(spans)

@pytest.mark.integration()
def test_gemini_chat_sample(setup):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    chat = client.chats.create(model="gemini-2.5-flash")

    response = chat.send_message("I have 2 dogs in my house.")
    print(response.text)

    response = chat.send_message("How many paws are in my house?")
    print(response.text)

    for message in chat.get_history():
        print(f'role - {message.role}', end=": ")
        print(message.parts[0].text)
    time.sleep(5)
    print(response.text)
    spans = custom_exporter.get_captured_spans()
    check_span(spans)

def check_span(spans):
    found_workflow_span = False
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
                span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.gemini"
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gemini-2.5-flash"
            assert span_attributes["entity.2.type"] == "model.llm.gemini-2.5-flash"

            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "workflow":
            found_workflow_span = True
    assert found_workflow_span

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0x3d2eaae04c01d8949140ca5e6c2eef8e",
#         "span_id": "0x3b498a0c89fc2fd2",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9b91e88a04606218",
#     "start_time": "2025-06-25T14:01:43.849583Z",
#     "end_time": "2025-06-25T14:01:47.789155Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_gemini_metamodel_sample.py:27",
#         "workflow.name": "gemini_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.gemini",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/",
#         "entity.2.name": "gemini-2.5-flash",
#         "entity.2.type": "model.llm.gemini-2.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-06-25T14:01:47.789155Z",
#             "attributes": {
#                 "input": [
#                     "{'system': 'You are a cat. Your name is Neko.'}",
#                     "{'input': 'Hello there'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-06-25T14:01:47.789155Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": [
#                       "{'model': \"*Mrow?*\\n\\nI blink a slow, deliberate blink, one golden eye opening to peer at you from my sunbeam-warmed spot on the windowsill. My tail gives a lazy flick, just the very tip. It's a nice sunbeam.\"}"
#                ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-06-25T14:01:47.789155Z",
#             "attributes": {
#                 "completion_tokens": 30,
#                 "prompt_tokens": 15,
#                 "total_tokens": 357
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x3d2eaae04c01d8949140ca5e6c2eef8e",
#         "span_id": "0x9b91e88a04606218",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-06-25T14:01:43.849583Z",
#     "end_time": "2025-06-25T14:01:47.789155Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\tests\\integration\\test_gemini_metamodel_sample.py:27",
#         "span.type": "workflow",
#         "entity.1.name": "gemini_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0x6ec9add2d431c024246101e05012ec3d",
#         "span_id": "0xa8f967d5e19b16b0",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x3e6ee0967e2491eb",
#     "start_time": "2025-06-25T14:01:52.804929Z",
#     "end_time": "2025-06-25T14:02:02.135250Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\checkvenv\\Lib\\site-packages\\google\\genai\\chats.py:259",
#         "workflow.name": "gemini_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.gemini",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/",
#         "entity.2.name": "gemini-2.5-flash",
#         "entity.2.type": "model.llm.gemini-2.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-06-25T14:02:02.134205Z",
#             "attributes": {
#                 "input": [
#                     "{'user': 'I have 2 dogs in my house.'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-06-25T14:02:02.135250Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": [
#                       "{'model': 'Oh, how wonderful! Dogs bring so much joy and fun to a home.\\n\\nDo you want to tell me anything about them, like their names or breeds?'}"
#                ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-06-25T14:02:02.135250Z",
#             "attributes": {
#                 "completion_tokens": 95,
#                 "prompt_tokens": 10,
#                 "total_tokens": 1301
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x6ec9add2d431c024246101e05012ec3d",
#         "span_id": "0x3e6ee0967e2491eb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-06-25T14:01:52.804929Z",
#     "end_time": "2025-06-25T14:02:02.135250Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\checkvenv\\Lib\\site-packages\\google\\genai\\chats.py:259",
#         "span.type": "workflow",
#         "entity.1.name": "gemini_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "google.genai.models.Models",
#     "context": {
#         "trace_id": "0xa21354200bd96d48a85b3a62e498234b",
#         "span_id": "0xdcc5d131ade08cfb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9d521b9e427234f4",
#     "start_time": "2025-06-25T14:02:02.136262Z",
#     "end_time": "2025-06-25T14:02:05.183003Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\checkvenv\\Lib\\site-packages\\google\\genai\\chats.py:259",
#         "workflow.name": "gemini_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.gemini",
#         "entity.1.inference_endpoint": "https://generativelanguage.googleapis.com/",
#         "entity.2.name": "gemini-2.5-flash",
#         "entity.2.type": "model.llm.gemini-2.5-flash",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-06-25T14:02:05.183003Z",
#             "attributes": {
#                 "input": [
#                     "{'user': 'I have 2 dogs in my house.'}",
#                     "{'model': \"That's wonderful! Dogs bring so much joy to a home.\\n\\nDo you want to share anything about them? Like:\\n*   **What are their names?**\\n*   **What kind of dogs are they?**\\n*   **How old are they?**\\n*   **Or are you just sharing that fact?**\\n\\nI'm here if you have any questions about dog care, training, or just want to chat about them!\"}",
#                     "{'user': 'How many paws are in my house?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-06-25T14:02:05.183003Z",
#             "attributes": {
#                 "status": "success",
#                 "status_code": "success",
#                 "response": [
                #     "{'model': \"*Mrow?*\\n\\nI blink a slow, deliberate blink, one golden eye opening to peer at you from my sunbeam-warmed spot on the windowsill. My tail gives a lazy flick, just the very tip. It's a nice sunbeam.\"}"
                # ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-06-25T14:02:05.183003Z",
#             "attributes": {
#                 "completion_tokens": 31,
#                 "prompt_tokens": 115,
#                 "total_tokens": 475
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xa21354200bd96d48a85b3a62e498234b",
#         "span_id": "0x9d521b9e427234f4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-06-25T14:02:02.136262Z",
#     "end_time": "2025-06-25T14:02:05.183003Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "C:\\Users\\BHLP0106\\Desktop\\clone\\monocle\\checkvenv\\Lib\\site-packages\\google\\genai\\chats.py:259",
#         "span.type": "workflow",
#         "entity.1.name": "gemini_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "gemini_app_1"
#         },
#         "schema_url": ""
#     }
# }