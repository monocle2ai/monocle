import logging
import time
import os

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL")

logger = logging.getLogger(__name__)
@pytest.fixture(scope="module")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="haystack_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[]
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_haystack_anthropic_sample(setup):
    generator = AnthropicChatGenerator(model=ANTHROPIC_MODEL,
                                       generation_kwargs={
                                           "max_tokens": 1000,
                                           "temperature": 0.7,
                                       })

    messages = [ChatMessage.from_system("You are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]
    response = generator.run(messages=messages)
    time.sleep(5)
    logger.info(response)

    spans = setup.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes

            assert span_attributes["entity.1.type"] == "inference.anthropic"
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == ANTHROPIC_MODEL 
            assert span_attributes["entity.2.type"] == "model.llm." + ANTHROPIC_MODEL

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if not span.parent and span.name == "workflow":
            assert span_attributes["entity.1.name"] == "haystack_app_1"
            assert span_attributes["entity.1.type"] == "workflow.haystack"

# {
#     "name": "anthropic.resources.messages.messages.Messages",
#     "context": {
#         "trace_id": "0x32c6b7230cbc97c536cafa6c3f96a2f0",
#         "span_id": "0x927160c89f837706",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x842cccc6c9e45438",
#     "start_time": "2025-05-01T17:01:12.597476Z",
#     "end_time": "2025-05-01T17:01:19.020999Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "workflow.name": "haystack_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
#     "context": {
#         "trace_id": "0x32c6b7230cbc97c536cafa6c3f96a2f0",
#         "span_id": "0x842cccc6c9e45438",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x67061c9252e5b84d",
#     "start_time": "2025-05-01T17:01:12.596476Z",
#     "end_time": "2025-05-01T17:01:19.024104Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "workflow.name": "haystack_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.anthropic",
#         "entity.1.inference_endpoint": "https://api.anthropic.com",
#         "entity.2.name": "claude-3-5-sonnet-20240620",
#         "entity.2.type": "model.llm.claude-3-5-sonnet-20240620",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-05-01T17:01:19.024104Z",
#             "attributes": {
#                 "input": [
#                     "{'system': 'You are a helpful, respectful and honest assistant'}",
#                     "{'user': \"What's Natural Language Processing?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-05-01T17:01:19.024104Z",
#             "attributes": {
#                 "response": [
#                     "Natural Language Processing (NLP) is a branch of artificial intelligence (AI) that focuses on the interaction between computers and human language. It involves the ability of computers to understand, interpret, generate, and manipulate human language in a way that is both meaningful and useful.\n\nKey aspects of NLP include:\n\n1. Text analysis: Breaking down and understanding the structure of written language.\n\n2. Speech recognition: Converting spoken language into text.\n\n3. Machine translation: Translating text from one language to another.\n\n4. Sentiment analysis: Determining the emotional tone behind words.\n\n5. Named entity recognition: Identifying and classifying named entities (e.g., person names, organizations) in text.\n\n6. Text summarization: Creating concise summaries of longer texts.\n\n7. Question answering: Providing accurate responses to human queries.\n\n8. Text generation: Creating human-like text based on input or prompts.\n\nNLP applications are widespread and include:\n\n- Virtual assistants (like Siri or Alexa)\n- Chatbots for customer service\n- Email filters and spam detection\n- Language translation services\n- Autocomplete and predictive text features\n\nNLP combines elements from linguistics, computer science, and machine learning to create systems that can process and understand human language in a way that is both meaningful and useful for various applications."
#                 ]
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-05-01T17:01:19.024104Z",
#             "attributes": {
#                 "completion_tokens": 294,
#                 "prompt_tokens": 23,
#                 "total_tokens": 317
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x32c6b7230cbc97c536cafa6c3f96a2f0",
#         "span_id": "0x67061c9252e5b84d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-05-01T17:01:12.596476Z",
#     "end_time": "2025-05-01T17:01:19.024104Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "monocle_apptrace.language": "python",
#         "span.type": "workflow",
#         "entity.1.name": "haystack_app_1",
#         "entity.1.type": "workflow.haystack",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
