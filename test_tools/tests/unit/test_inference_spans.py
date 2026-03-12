import pytest
from monocle_test_tools import MonocleValidator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, StatusCode, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanKind


def create_test_span(name, attributes, has_error=False, has_warning=False):
    """Helper function to create a test span."""
    tracer_provider = TracerProvider(resource=Resource(attributes={}))
    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)

        if has_error:
            span.set_status(StatusCode.ERROR, "Test error")

        if has_warning:
            span.add_event("metadata", attributes={"finish_type": "warning"})

        # Get the span object
        sdk_span = span

    return sdk_span._readable_span()


def test_get_inference_spans_no_filters():
    """Test _get_inference_spans without filtering for errors or warnings."""
    validator = MonocleValidator()

    # Create test spans - mix of inference spans with and without errors/warnings
    span1 = create_test_span(
        "inference1", {"span.type": "inference"}, has_error=False, has_warning=False
    )
    span2 = create_test_span(
        "inference2", {"span.type": "inference"}, has_error=True, has_warning=False
    )
    span3 = create_test_span(
        "inference3",
        {"span.type": "inference.framework"},
        has_error=False,
        has_warning=True,
    )

    # Export spans to memory exporter
    validator.memory_exporter.export([span1, span2, span3])

    # Test without filtering - should return all inference spans
    inference_spans = validator._get_inference_spans(
        expect_errors=False, expect_warnings=False
    )

    # Should only return span1 (no errors, no warnings)
    assert len(inference_spans) == 1
    assert inference_spans[0].name == "inference1"

    validator.cleanup()


def test_get_inference_spans_with_error_filter():
    """Test _get_inference_spans filtering for spans with errors."""
    validator = MonocleValidator()

    # Create test spans
    span1 = create_test_span(
        "inference1", {"span.type": "inference"}, has_error=False, has_warning=False
    )
    span2 = create_test_span(
        "inference2", {"span.type": "inference"}, has_error=True, has_warning=False
    )

    # Export spans to memory exporter
    validator.memory_exporter.export([span1, span2])

    # Test filtering for errors
    inference_spans = validator._get_inference_spans(
        expect_errors=True, expect_warnings=False
    )

    # Should only return span2 (has error, no warning)
    assert len(inference_spans) == 1
    assert inference_spans[0].name == "inference2"

    validator.cleanup()


def test_get_inference_spans_with_warning_filter():
    """Test _get_inference_spans filtering for spans with warnings."""
    validator = MonocleValidator()

    # Create test spans
    span1 = create_test_span(
        "inference1", {"span.type": "inference"}, has_error=False, has_warning=False
    )
    span2 = create_test_span(
        "inference2",
        {"span.type": "inference.framework"},
        has_error=False,
        has_warning=True,
    )

    # Export spans to memory exporter
    validator.memory_exporter.export([span1, span2])

    # Test filtering for warnings
    inference_spans = validator._get_inference_spans(
        expect_errors=False, expect_warnings=True
    )

    # Should only return span2 (no error, has warning)
    assert len(inference_spans) == 1
    assert inference_spans[0].name == "inference2"

    validator.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
