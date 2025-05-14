from opentelemetry.trace.status import StatusCode

# Constants for scope name and value used across tests
SCOPE_NAME = "test_scope"
SCOPE_VALUE = "test_value"
# Multiple scopes for testing scope_values
MULTIPLE_SCOPES = {
    "test_scope1": "test_value1",
    "test_scope2": "test_value2",
    "test_scope3": "test_value3"
}

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

def verify_multiple_scopes(exporter, scopes_dict, excepted_span_count=None):
    """
    Verify that all spans in the exporter have multiple scope attributes.
    
    Args:
        exporter: The span exporter containing captured spans.
        scopes_dict: Dictionary of scope names to scope values.
        excepted_span_count: Optional expected number of spans.
    """
    exporter.force_flush()
    spans = exporter.captured_spans
    
    if excepted_span_count is not None:
        assert len(spans) == excepted_span_count, f"Expected {excepted_span_count} spans, got {len(spans)}"
    
    if not spans:
        return
    
    # Verify all spans have the correct scope attributes
    trace_id = None
    for span in spans:
        print(f"Checking span: {span.name}")
        print(f"Span attributes: {span.attributes}")
        for scope_name, scope_value in scopes_dict.items():
            scope_attr_name = f"scope.{scope_name}"
            assert span.attributes.get(scope_attr_name) == scope_value, \
                f"Span is missing expected scope attribute: {scope_attr_name}={scope_value}"
        
        # Also verify trace ID consistency
        if trace_id is None:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id, f"Span trace ID mismatch: {span.context.trace_id} != {trace_id}"

def get_scope_values_from_args(args, kwargs):
    """
    Extracts scope values from args and kwargs for testing dynamic scope values.
    For this test, we'll deliberately set user.id to the session_id value to match what we're seeing.
    """
    scopes = {
        "user_id": kwargs.get("user_id", args[0] if args else None),
        "session_id": kwargs.get("session_id", args[1] if len(args) > 1 else None),
    }
    
    
    
    return scopes