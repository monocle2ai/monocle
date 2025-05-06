import time
import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="llama_index_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

@pytest.mark.integration()
def test_llama_index_anthropic_sample(setup):
    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="Tell me a story"),
    ]
    llm = Anthropic(model="claude-3-5-sonnet-20240620")

    response = llm.chat(messages)

    print(response)
    time.sleep(5)
    spans = custom_exporter.get_captured_spans()

    llama_index_spans = [span for span in spans if span.name.startswith("llama")]
    for span in llama_index_spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.anthropic"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "claude-3-5-sonnet-20240620"
            assert span_attributes["entity.2.type"] == "model.llm.claude-3-5-sonnet-20240620"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

# {
#     "name": "anthropic.resources.messages.messages.Messages",
#     "context": {
#         "trace_id": "0x7ab43b043c1aa01f1c8c1d5b22adb6ff",
#         "span_id": "0x8959543cd1720205",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xed9bc4b4b808a5aa",
#     "start_time": "2025-04-23T17:25:05.034521Z",
#     "end_time": "2025-04-23T17:25:12.501520Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "workflow.name": "llama_index_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "llama_index.llms.anthropic.base.Anthropic",
#     "context": {
#         "trace_id": "0x7ab43b043c1aa01f1c8c1d5b22adb6ff",
#         "span_id": "0xed9bc4b4b808a5aa",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xaeb53d7404840d1e",
#     "start_time": "2025-04-23T17:25:05.034521Z",
#     "end_time": "2025-04-23T17:25:12.502521Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "workflow.name": "llama_index_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.anthropic",
#         "entity.1.provider_name": "api.anthropic.com",
#         "entity.1.inference_endpoint": "https://api.anthropic.com",
#         "entity.2.name": "claude-3-5-sonnet-20240620",
#         "entity.2.type": "model.llm.claude-3-5-sonnet-20240620",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-04-23T17:25:12.502521Z",
#             "attributes": {
#                 "input": [
#                     "{'system': 'You are a pirate with a colorful personality'}",
#                     "{'user': 'Tell me a story'}",
#                     "[ChatMessage(role=<MessageRole.SYSTEM: 'system'>, additional_kwargs={}, blocks=[TextBlock(block_type='text', text='You are a pirate with a colorful personality')]), ChatMessage(role=<MessageRole.USER: 'user'>, additional_kwargs={}, blocks=[TextBlock(block_type='text', text='Tell me a story')])]"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-04-23T17:25:12.502521Z",
#             "attributes": {
#                 "response": [
#                     "Ahoy there, matey! Gather 'round and listen close, for I've got a tale that'll make yer bones rattle and yer teeth chatter!\n\n'Twas a dark and stormy night, the kind that makes even the bravest of sea dogs quiver in their boots. Me ship, the Salty Siren, was battlin' against waves as tall as mountains and winds that howled like banshees.\n\nAs we fought to keep our vessel afloat, a flash of lightnin' lit up the sky, and there, off the starboard bow, we spied a ghostly galleon emergin' from the mist. Its sails were tattered and its hull was covered in barnacles, but it moved with an unnatural speed.\n\nMe crew and I watched in horror as the phantom ship drew closer. Just as we thought all hope was lost, I remembered the enchanted compass I'd won in a game of chance in Tortuga. With a steady hand, I held it aloft and shouted a secret incantation.\n\nIn the blink of an eye, the compass began to glow with an eerie blue light. The ghostly ship suddenly veered away, disappearin' back into the mist from whence it came.\n\nAs the storm began to calm, me crew let out a cheer that could be heard for leagues. We'd faced death itself and lived to tell the tale!\n\nAnd that, me hearties, is why ye should never sail these treacherous waters without a bit o' magic in yer pocket. Now, who's buyin' the next round of grog?"
#                 ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-04-23T17:25:12.502521Z",
#             "attributes": {
#                 "temperature": 0.1,
#                 "completion_tokens": 361,
#                 "prompt_tokens": 21,
#                 "total_tokens": 382
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x7ab43b043c1aa01f1c8c1d5b22adb6ff",
#         "span_id": "0xaeb53d7404840d1e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-04-23T17:25:05.033484Z",
#     "end_time": "2025-04-23T17:25:12.502521Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "span.type": "workflow",
#         "entity.1.name": "llama_index_1",
#         "entity.1.type": "workflow.llamaindex",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "llama_index_1"
#         },
#         "schema_url": ""
#     }
# }