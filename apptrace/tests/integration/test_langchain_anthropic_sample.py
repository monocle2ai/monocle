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
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL")


@pytest.fixture(scope="module")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="langchain_app_1",
            span_processors=[BatchSpanProcessor(custom_exporter)],
            wrapper_methods=[],
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_langchain_anthropic_sample(setup):
    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    ai_answer = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    time.sleep(5)

    spans = setup.get_captured_spans()
    assert len(spans) > 0, "No spans captured for the LangChain Anthropic sample"

    workflow_span = None

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
            r"^\{\"human\": \".+\"\}$",  # Pattern for human message
        ],
        output_pattern=r"^\{\"ai\": \".+\"\}$",  # Pattern for AI response
        metadata_requirements={
            "temperature": float,
            "completion_tokens": int,
            "prompt_tokens": int,
            "total_tokens": int,
        },
    )

    workflow_span = find_span_by_type(spans, "workflow")

    assert workflow_span is not None, "Expected to find workflow span"

    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "langchain_app_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.langchain"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0xfba3bb15f398c656cd610b7e65904df0",
#         "span_id": "0x8d5cbdb96a0bba09",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x44e42513f2ba1ec8",
#     "start_time": "2025-07-13T10:06:49.906855Z",
#     "end_time": "2025-07-13T10:06:49.907280Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3045",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
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
#     "name": "anthropic.resources.messages.messages.Messages",
#     "context": {
#         "trace_id": "0xfba3bb15f398c656cd610b7e65904df0",
#         "span_id": "0x05236cd6b687db1d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xd90a1f9136d343f4",
#     "start_time": "2025-07-13T10:06:49.928339Z",
#     "end_time": "2025-07-13T10:06:51.466128Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_anthropic/chat_models.py:1204",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.modelapi"
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
#     "name": "langchain_anthropic.chat_models.ChatAnthropic",
#     "context": {
#         "trace_id": "0xfba3bb15f398c656cd610b7e65904df0",
#         "span_id": "0xd90a1f9136d343f4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x44e42513f2ba1ec8",
#     "start_time": "2025-07-13T10:06:49.907449Z",
#     "end_time": "2025-07-13T10:06:51.468539Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
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
#             "timestamp": "2025-07-13T10:06:51.468475Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant that translates English to German.\"}",
#                     "{\"human\": \"I love programming.\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T10:06:51.468515Z",
#             "attributes": {
#                 "response": "{\"ai\": \"Here's the German translation:\\n\\nIch liebe Programmieren.\"}",
#                 "finish_reason": "end_turn",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T10:06:51.468528Z",
#             "attributes": {
#                 "temperature": 0.0,
#                 "completion_tokens": 18,
#                 "prompt_tokens": 23,
#                 "total_tokens": 41
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
#     "name": "langchain_core.runnables.base.RunnableSequence",
#     "context": {
#         "trace_id": "0xfba3bb15f398c656cd610b7e65904df0",
#         "span_id": "0x44e42513f2ba1ec8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x32e36a946969e172",
#     "start_time": "2025-07-13T10:06:49.902694Z",
#     "end_time": "2025-07-13T10:06:51.468607Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_anthropic_sample.py:50",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
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
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xfba3bb15f398c656cd610b7e65904df0",
#         "span_id": "0x32e36a946969e172",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-13T10:06:49.902635Z",
#     "end_time": "2025-07-13T10:06:51.468622Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_anthropic_sample.py:50",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.langchain",
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
