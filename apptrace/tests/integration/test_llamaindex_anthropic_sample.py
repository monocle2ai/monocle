import logging
import time
import os

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL")

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="llama_index_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_llama_index_anthropic_sample(setup):
    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="Tell me a story in 10 words"),
    ]
    llm = Anthropic(model=ANTHROPIC_MODEL)

    response = llm.chat(messages)

    logger.info(response)
    time.sleep(5)
    spans = setup.get_captured_spans()

    assert len(spans) > 0, "No spans captured for the LangChain Anthropic sample"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        # Also check for inference.framework spans
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected to find at least one inference span"

    # Verify each inference span
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.anthropic",
            model_name=ANTHROPIC_MODEL,
            model_type="model.llm." + ANTHROPIC_MODEL,
            check_metadata=True,
            check_input_output=True,
        )
    assert (
        len(inference_spans) == 1
    ), "Expected exactly one inference span for the LLM call"

    # Validate events using the generic function with regex patterns
    validate_inference_span_events(
        span=inference_spans[0],
        expected_event_count=3,
        input_patterns=[
            r"^\{\"system\": \".+\"\}$",  # Pattern for system message
            r"^\{\"user\": \".+\"\}$",  # Pattern for user message
        ],
        output_pattern=r"^\{\"assistant\": \".+\"\}$",  # Pattern for assistant response
        metadata_requirements={
            "temperature": float,
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int,
        },
    )
    # pick the last span as there is are two workflow spans:
    # one for openai embedding one for llamaindex query
    workflow_span = find_span_by_type(spans, "workflow")

    assert workflow_span is not None, "Expected to find workflow span"

    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "llama_index_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.llamaindex"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "anthropic.resources.messages.messages.Messages",
#     "context": {
#         "trace_id": "0xa3596e9a520629d8593e58a3fc7f7906",
#         "span_id": "0xb20d818676f27716",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x6dd852b403c2eedf",
#     "start_time": "2025-07-02T16:15:03.906536Z",
#     "end_time": "2025-07-02T16:15:06.464481Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/llama_index/llms/anthropic/base.py:352",
#         "workflow.name": "llama_index_1",
#         "span.type": "inference.modelapi"
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
#         "trace_id": "0xa3596e9a520629d8593e58a3fc7f7906",
#         "span_id": "0x6dd852b403c2eedf",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x0ff2fb9e9cada346",
#     "start_time": "2025-07-02T16:15:03.905191Z",
#     "end_time": "2025-07-02T16:15:06.464850Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_llamaindex_anthropic_sample.py:31",
#         "workflow.name": "llama_index_1",
#         "span.type": "inference.framework",
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
#             "timestamp": "2025-07-02T16:15:06.464806Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a pirate with a colorful personality\"}",
#                     "{\"user\": \"Tell me a story in 10 words\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-02T16:15:06.464829Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"Arrr, ye scurvy dog! Here be a tale in ten words:\\n\\nTreasure map led to cursed gold. Crew mutinied. Captain's revenge sweet.\"}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-02T16:15:06.464845Z",
#             "attributes": {
#                 "temperature": 0.1,
#                 "completion_tokens": 42,
#                 "prompt_tokens": 26,
#                 "total_tokens": 68
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
#         "trace_id": "0xa3596e9a520629d8593e58a3fc7f7906",
#         "span_id": "0x0ff2fb9e9cada346",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-02T16:15:03.905133Z",
#     "end_time": "2025-07-02T16:15:06.464857Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_llamaindex_anthropic_sample.py:31",
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
