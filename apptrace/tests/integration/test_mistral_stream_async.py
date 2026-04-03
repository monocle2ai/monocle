import asyncio
import logging
import os

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from common.helpers import find_span_by_type, find_spans_by_type, verify_inference_span
from mistralai.client import Mistral
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

logger = logging.getLogger(__name__)

MODEL = "mistral-large-latest"


@pytest.fixture(scope="function")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter),
    ]

    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="generic_mistral_stream_async",
            span_processors=span_processors,
            wrapper_methods=[],
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_mistral_stream_async(setup):
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    response = await client.chat.stream_async(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "How far is the moon from earth? Answer with the distance in km only.",
            },
        ],
    )

    chunks = []
    texts = []
    async for chunk in response:
        chunks.append(chunk)
        if chunk.data.choices[0].delta.content is not None:
            texts.append(chunk.data.choices[0].delta.content)

    full_response = "".join(texts).strip()
    assert len(chunks) > 0, "Expected streaming chunks"
    assert len(full_response) > 0, "Expected non-empty streamed response"

    await asyncio.sleep(5)

    spans = setup.get_captured_spans()
    logger.info(f"Captured {len(spans)} spans")
    assert len(spans) > 0, "No spans captured"

    inference_spans = find_spans_by_type(spans, "inference")
    if not inference_spans:
        inference_spans = find_spans_by_type(spans, "inference.framework")

    assert len(inference_spans) > 0, "Expected at least one inference span"

    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.mistral",
            model_name=MODEL,
            model_type=f"model.llm.{MODEL}",
            check_metadata=False,
            check_input_output=True,
        )
        assert span.attributes.get("span.subtype") in ["turn_end", "tool_call", "delegation"]

    workflow_span = find_span_by_type(spans, "workflow")
    assert workflow_span is not None, "Expected workflow span"
    assert workflow_span.attributes["entity.1.name"] == "generic_mistral_stream_async"
    assert workflow_span.attributes["entity.1.type"] == "workflow.mistral"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
