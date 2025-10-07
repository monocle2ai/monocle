import os
import pytest
import time
import asyncio
from mistralai import Mistral, models
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from common.helpers import (
    find_span_by_type,
    find_spans_by_type,
    validate_inference_span_events,
    verify_inference_span,
)
from monocle_apptrace.instrumentation.metamodel.mistral._helper import (
    map_mistral_finish_reason_to_finish_type,
)
 
custom_exporter = CustomConsoleSpanExporter()
 
@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="generic_mistral_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[],
    )
 
@pytest.fixture(autouse=True)
def clear_spans():
    custom_exporter.reset()
    yield
 
@pytest.mark.integration()
def test_mistral_invalid_api_key(setup):
    try:
        client = Mistral(api_key="invalid_key_123")
        client.chat.complete(
            model="mistral-small",
            messages=[{"role": "user", "content": "test"}],
        )
    except models.SDKError as e:
        # e.status_code is correct
        assert e.status_code == 401
        # The response body contains the actual "Unauthorized" text
        assert '"Unauthorized"' in e.body
 
    # Wait for spans to be flushed
    time.sleep(5)
    spans = custom_exporter.get_captured_spans()
 
    for span in spans:
        if span.attributes.get("span.type") in ["inference", "inference.framework"]:
            events = [e for e in span.events if e.name == "data.output"]
            assert len(events) > 0
            # Span should have ERROR status
            assert span.status.status_code.value == 2
 
            # Current error_code in span is just "error"
            error_code = events[0].attributes.get("error_code")
            assert error_code == "error"
 
            response = events[0].attributes.get("response")
            assert response is None or response == ""
 
 
@pytest.mark.integration()
def test_mistral_api_streaming_sync(setup):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
 
    # Start streaming response synchronously
    stream_response = client.chat.stream(
        model="mistral-small",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )
 
    # Collect streamed content and finish_reason
    collected_output = ""
    finish_reason = None
    for chunk in stream_response:
        if chunk.data.choices[0].delta.content is not None:
            collected_output += chunk.data.choices[0].delta.content
        if chunk.data.choices[0].finish_reason is not None:
            finish_reason = chunk.data.choices[0].finish_reason
 
    # Map finish_reason → finish_type
    mapped_finish_type = map_mistral_finish_reason_to_finish_type(finish_reason)
    print(f"Finish reason: {finish_reason}, mapped type: {mapped_finish_type}")
 
    # Give some time for spans to flush
    asyncio.run(asyncio.sleep(5))
 
    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")
 
    assert len(spans) > 0, "No spans captured"
 
    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        inference_spans = find_spans_by_type(spans, "inference.framework")
 
    assert len(inference_spans) > 0, "Expected at least one inference span"
 
    # Verify spans (no skip logic)
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.mistral",
            model_name="mistral-small",
            model_type="model.llm.mistral-small",
            check_metadata=True,
            check_input_output=True,
        )
 
    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None, "Expected workflow span"
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "generic_mistral_1"
    assert workflow_span.attributes["workflow.name"] == "generic_mistral_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.mistral"
 
 
@pytest.mark.integration()
@pytest.mark.asyncio
async def test_mistral_api_streaming_async(setup):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
 
    # Start streaming response asynchronously
    response = await client.chat.stream_async(
        model="mistral-small",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an americano?"},
        ],
    )
 
    # Collect streamed content and finish_reason
    collected_output = ""
    finish_reason = None
    async for chunk in response:
        if chunk.data.choices[0].delta.content is not None:
            collected_output += chunk.data.choices[0].delta.content
        if chunk.data.choices[0].finish_reason is not None:
            finish_reason = chunk.data.choices[0].finish_reason
 
    # Map finish_reason → finish_type
    mapped_finish_type = map_mistral_finish_reason_to_finish_type(finish_reason)
    print(f"Finish reason: {finish_reason}, mapped type: {mapped_finish_type}")
 
    # Give some time for spans to flush
    await asyncio.sleep(5)
 
    spans = custom_exporter.get_captured_spans()
    print(f"Captured {len(spans)} spans")
 
    assert len(spans) > 0, "No spans captured"
 
    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        inference_spans = find_spans_by_type(spans, "inference.framework")
 
    assert len(inference_spans) > 0, "Expected at least one inference span"
 
    # Verify spans (no skip logic)
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.mistral",
            model_name="mistral-small",
            model_type="model.llm.mistral-small",
            check_metadata=True,
            check_input_output=True,
        )
 
    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None, "Expected workflow span"
    assert workflow_span.attributes["span.type"] == "workflow"
    assert workflow_span.attributes["entity.1.name"] == "generic_mistral_1"
    assert workflow_span.attributes["workflow.name"] == "generic_mistral_1"
    assert workflow_span.attributes["entity.1.type"] == "workflow.mistral"
if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])