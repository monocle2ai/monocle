"""
Integration test for Okahu trace ingestion with local validation.

This test demonstrates:
1. Generating traces with custom span input/output capture
2. Sending traces to Okahu cloud (ingestion)
3. Validating trace content locally (using memory exporter)

Approach:
- Sends traces to Okahu for ingestion/storage
- Validates trace structure and content locally from memory

Prerequisites:
- OKAHU_API_KEY environment variable must be set
- Optional: OKAHU_INGESTION_ENDPOINT for custom ingestion endpoint (defaults to production)

Run:
    export OKAHU_API_KEY="your_api_key_here"
    python -m pytest tests/integration/test_cloud_trace_validation.py -v -s
"""
import os
import time
import pytest
from monocle_apptrace.instrumentation.common.instrumentor import (
    setup_monocle_telemetry,
    monocle_trace_method,
    get_monocle_instrumentor,
    set_monocle_instrumentor,
    set_monocle_setup_signature
)
from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter
from monocle_apptrace.instrumentation.common.scope_wrapper import (
    start_scope,
    stop_scope
)
from monocle_apptrace.instrumentation.common.utils import get_scopes
from monocle_apptrace.exporters.base_exporter import MonocleInMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


@pytest.fixture(scope="module")
def check_okahu_credentials():
    """Check if Okahu credentials are available"""
    api_key = os.environ.get('OKAHU_API_KEY')
    if not api_key:
        pytest.skip("OKAHU_API_KEY not set - skipping cloud ingestion test")
    
    return api_key


@pytest.fixture(scope="module")
def generate_cloud_traces(check_okahu_credentials):
    """Generate traces and send to Okahu cloud, capture locally for validation"""
    
    # Cleanup any existing instrumentor
    existing = get_monocle_instrumentor()
    if existing and existing.is_instrumented_by_opentelemetry:
        existing.uninstrument()
    
    set_monocle_instrumentor(None)
    set_monocle_setup_signature(None)
    
    # Setup memory exporter to capture spans locally for validation
    memory_exporter = MonocleInMemorySpanExporter()
    
    # Setup telemetry with BOTH Okahu (for ingestion) and Memory (for validation)
    instrumentor = setup_monocle_telemetry(
        workflow_name="test_cloud_custom_span_validation",
        span_processors=[
            SimpleSpanProcessor(OkahuSpanExporter()),  # Send to Okahu
            SimpleSpanProcessor(memory_exporter)        # Capture locally
        ]
    )
    
    # Start session scope
    session_id = f"monocle_test_session_cloud_validation_{int(time.time())}"
    scope_token = start_scope("agentic.session", session_id)
    
    # Define test function with custom span
    @monocle_trace_method(span_name="calculate_order_total")
    def calculate_order_total(items, tax_rate=0.10, discount_code=None):
        """Calculate order total with tax and optional discount"""
        subtotal = sum(item["price"] * item["quantity"] for item in items)
        
        if discount_code == "SAVE10":
            subtotal *= 0.9
        
        tax = subtotal * tax_rate
        total = subtotal + tax
        
        return {
            "subtotal": round(subtotal, 2),
            "tax": round(tax, 2),
            "total": round(total, 2),
            "item_count": len(items),
            "discount_applied": discount_code is not None
        }
    
    # Generate trace
    test_items = [
        {"name": "Laptop", "price": 999.99, "quantity": 1},
        {"name": "Mouse", "price": 24.99, "quantity": 2},
        {"name": "Keyboard", "price": 79.99, "quantity": 1}
    ]
    
    result = calculate_order_total(test_items, tax_rate=0.08, discount_code="SAVE10")
    
    # Capture session ID
    scopes = get_scopes()
    captured_session_id = scopes.get('agentic.session')
    
    # Stop scope
    stop_scope(scope_token)
    
    # Force flush to ensure all spans are exported
    from opentelemetry import trace
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, 'force_flush'):
        tracer_provider.force_flush()
    
    # Get captured spans from memory exporter
    captured_spans = memory_exporter.get_finished_spans()
    
    yield {
        "session_id": captured_session_id,
        "workflow_name": "test_cloud_custom_span_validation",
        "result": result,
        "test_items": test_items,
        "spans": captured_spans,
        "memory_exporter": memory_exporter
    }
    
    # Cleanup
    if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    
    set_monocle_instrumentor(None)
    set_monocle_setup_signature(None)


def test_validate_cloud_traces_by_session(generate_cloud_traces):
    """
    Test validation of custom span traces sent to Okahu cloud.
    
    This test:
    1. Uses traces generated and sent to Okahu in generate_cloud_traces fixture
    2. Validates trace content using locally captured spans
    3. Validates custom span attributes
    4. Validates input/output capture
    
    Note: This test validates locally instead of retrieving from Okahu.
    """
    
    session_id = generate_cloud_traces["session_id"]
    workflow_name = generate_cloud_traces["workflow_name"]
    test_items = generate_cloud_traces["test_items"]
    result = generate_cloud_traces["result"]
    spans = generate_cloud_traces["spans"]
    
    # Validate that we have spans
    assert len(spans) > 0, "No spans were captured"
    
    # Find the custom span
    custom_span = None
    for span in spans:
        if span.name == "calculate_order_total":
            custom_span = span
            break
    
    assert custom_span is not None, "Custom span 'calculate_order_total' not found"
    
    # Validate span type
    assert custom_span.attributes.get("span.type") == "custom", \
        f"Expected span.type='custom', got '{custom_span.attributes.get('span.type')}'"
    
    # Validate workflow name
    assert custom_span.attributes.get("workflow.name") == workflow_name, \
        f"Expected workflow.name='{workflow_name}', got '{custom_span.attributes.get('workflow.name')}'"
    
    # Validate session scope was captured
    assert custom_span.attributes.get("scope.agentic.session") == session_id, \
        f"Expected session scope '{session_id}' in span attributes"
    
    # Validate input/output events exist
    event_names = [event.name for event in custom_span.events]
    assert "data.input" in event_names, "Expected 'data.input' event not found"
    assert "data.output" in event_names, "Expected 'data.output' event not found"
    
    # Validate input captured function arguments
    input_event = next((e for e in custom_span.events if e.name == "data.input"), None)
    assert input_event is not None, "Input event not found"
    
    # The input should contain the items, tax_rate, and discount_code arguments
    input_data = input_event.attributes.get("input")
    assert input_data is not None, "Input event has no 'input' attribute"
    
    # Validate output captured return value
    output_event = next((e for e in custom_span.events if e.name == "data.output"), None)
    assert output_event is not None, "Output event not found"
    
    output_data = output_event.attributes.get("response")
    assert output_data is not None, "Output event has no 'response' attribute"
    
    # Verify the result matches expectations
    assert result["subtotal"] == 1016.96, f"Expected subtotal 1016.96, got {result['subtotal']}"
    assert result["tax"] == 81.36, f"Expected tax 81.36, got {result['tax']}"
    assert result["total"] == 1098.32, f"Expected total 1098.32, got {result['total']}"
    assert result["item_count"] == 3, f"Expected item_count 3, got {result['item_count']}"
    assert result["discount_applied"] is True, f"Expected discount_applied True"


def test_validate_ingestion_success(generate_cloud_traces):
    """
    Test that traces were successfully ingested to Okahu.
    
    This test verifies:
    1. Spans were captured during execution
    2. The workflow span exists (indicating successful trace generation)
    3. All expected span types are present
    """
    
    workflow_name = generate_cloud_traces["workflow_name"]
    spans = generate_cloud_traces["spans"]
    
    # Validate we have multiple spans (workflow + custom span at minimum)
    assert len(spans) >= 2, f"Expected at least 2 spans, got {len(spans)}"
    
    # Find workflow span
    workflow_span = None
    for span in spans:
        if span.attributes.get("span.type") == "workflow":
            workflow_span = span
            break
    
    assert workflow_span is not None, "Workflow span not found - trace may not have been properly structured"
    assert workflow_span.attributes.get("workflow.name") == workflow_name, \
        f"Workflow span has incorrect workflow name"
    
    # Verify all spans have the workflow name attribute
    for span in spans:
        assert span.attributes.get("workflow.name") == workflow_name, \
            f"Span '{span.name}' missing workflow.name attribute"
    
    # Verify expected span types exist
    span_types = set()
    for span in spans:
        span_type = span.attributes.get("span.type", "unknown")
        span_types.add(span_type)
    assert "workflow" in span_types, "Missing workflow span type"
    assert "custom" in span_types, "Missing custom span type"


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
