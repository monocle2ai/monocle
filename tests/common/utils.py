from opentelemetry.trace.status import StatusCode

# Constants for scope name and value used across tests
SCOPE_NAME = "test_scope"
SCOPE_VALUE = "test_value"

def verify_traceID(exporter, excepted_span_count=None):
    """
    Verify that all spans in the exporter have the same trace ID.
    
    Args:
        exporter: The span exporter containing captured spans.
        excepted_span_count: Optional expected number of spans.
    """
    exporter.force_flush()
    spans = exporter.captured_spans
    
    if excepted_span_count is not None:
        assert len(spans) == excepted_span_count, f"Expected {excepted_span_count} spans, got {len(spans)}"
    
    if not spans:
        return
    
    # Get the trace ID from the first span
    trace_id = spans[0].context.trace_id
    
    # Verify all spans have the same trace ID
    for span in spans:
        assert span.context.trace_id == trace_id, f"Span trace ID mismatch: {span.context.trace_id} != {trace_id}"

def verify_scope(exporter, excepted_span_count=None):
    """
    Verify that all spans in the exporter have the same scope attribute.
    
    Args:
        exporter: The span exporter containing captured spans.
        excepted_span_count: Optional expected number of spans.
    """
    exporter.force_flush()
    spans = exporter.captured_spans
    
    if excepted_span_count is not None:
        assert len(spans) == excepted_span_count, f"Expected {excepted_span_count} spans, got {len(spans)}"
    
    if not spans:
        return
    
    # Verify all spans have the correct scope attribute
    trace_id = None
    for span in spans:
        assert span.attributes.get(f"scope.{SCOPE_NAME}") == SCOPE_VALUE, f"Span is missing expected scope attribute"
        
        # Also verify trace ID consistency
        if trace_id is None:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id, f"Span trace ID mismatch: {span.context.trace_id} != {trace_id}"