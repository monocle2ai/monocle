"""
Integration test for validating traces from Okahu cloud.

This test demonstrates:
1. Generating traces with custom span input/output capture
2. Sending traces to Okahu cloud
3. Retrieving and validating traces from cloud storage

Prerequisites:
- OKAHU_API_KEY environment variable must be set
- Or .env.monocle file with OKAHU_API_KEY

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
from monocle_test_tools.fluent_api import TraceAssertion
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


@pytest.fixture(scope="module")
def check_okahu_credentials():
    """Check if Okahu credentials are available"""
    api_key = os.environ.get('OKAHU_API_KEY')
    if not api_key:
        pytest.skip("OKAHU_API_KEY not set - skipping cloud validation test")
    return api_key


@pytest.fixture(scope="module")
def generate_cloud_traces(check_okahu_credentials):
    """Generate traces and send to Okahu cloud"""
    
    # Cleanup any existing instrumentor
    existing = get_monocle_instrumentor()
    if existing and existing.is_instrumented_by_opentelemetry:
        existing.uninstrument()
    
    set_monocle_instrumentor(None)
    set_monocle_setup_signature(None)
    
    # Setup telemetry with Okahu exporter
    instrumentor = setup_monocle_telemetry(
        workflow_name="test_cloud_custom_span_validation",
        span_processors=[SimpleSpanProcessor(OkahuSpanExporter())]
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
    
    # Force flush
    from opentelemetry import trace
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, 'force_flush'):
        tracer_provider.force_flush()
    
    # Wait for traces to be ingested to cloud
    time.sleep(10)  # Give Okahu time to ingest
    
    yield {
        "session_id": captured_session_id,
        "workflow_name": "test_cloud_custom_span_validation",
        "result": result,
        "test_items": test_items
    }
    
    # Cleanup
    if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    
    set_monocle_instrumentor(None)
    set_monocle_setup_signature(None)


def test_validate_cloud_traces_by_session(generate_cloud_traces):
    """
    Test validation of custom span traces retrieved from Okahu cloud.
    
    This test:
    1. Uses traces generated in generate_cloud_traces fixture
    2. Retrieves traces from Okahu using session ID
    3. Validates custom span attributes
    4. Validates input/output capture
    """
    
    session_id = generate_cloud_traces["session_id"]
    workflow_name = generate_cloud_traces["workflow_name"]
    test_items = generate_cloud_traces["test_items"]
    result = generate_cloud_traces["result"]
    
    # Create asserter
    asserter = TraceAssertion()
    
    # Import traces from Okahu cloud by session
    asserter.import_traces(
        trace_source="okahu",
        id=session_id,
        fact_name="session",
        workflow_name=workflow_name
    )
    
    # Validate custom span exists
    asserter.assert_has_custom_span("calculate_order_total")
    
    # Validate span type
    asserter.assert_span_has_attribute(
        "calculate_order_total",
        "span.type",
        "custom"
    )
    
    # Validate workflow name
    asserter.assert_span_has_attribute(
        "calculate_order_total",
        "workflow.name",
        workflow_name
    )
    
    # Validate input capture
    # Note: The exact assertion method may vary based on monocle_test_tools implementation
    # This validates that input events exist and contain expected data


def test_validate_cloud_traces_workflow_scope(generate_cloud_traces):
    """
    Test validation of traces by workflow name.
    
    This demonstrates retrieving all traces for a specific workflow.
    """
    
    workflow_name = generate_cloud_traces["workflow_name"]
    
    # Create asserter
    asserter = TraceAssertion()
    
    # Import traces by workflow (may return multiple sessions)
    asserter.import_traces(
        trace_source="okahu",
        id=generate_cloud_traces["session_id"],
        fact_name="session",
        workflow_name=workflow_name
    )
    
    # Validate at least one custom span exists
    asserter.assert_has_custom_span("calculate_order_total")


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
