"""
Integration test for monocle_trace_method decorator with input/output capture.
Creates actual trace files to show testers what custom spans produce.

Run test:
    python -m pytest tests/integration/test_custom_span_io.py -v -s
    
View trace files:
    ls -la .monocle/test_traces/
    cat .monocle/test_traces/monocle_trace_custom_span_*.json | python -m json.tool
"""
import json
import os
import glob
import time
import pytest
from monocle_apptrace.instrumentation.common.instrumentor import (
    setup_monocle_telemetry,
    monocle_trace_method,
)
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace


TEST_TRACE_DIR = ".monocle/test_traces"


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry with file exporter"""
    from monocle_apptrace.instrumentation.common.instrumentor import (
        get_monocle_instrumentor,
        set_monocle_instrumentor,
        set_monocle_setup_signature
    )
    
    # Cleanup any existing instrumentor
    existing = get_monocle_instrumentor()
    if existing and existing.is_instrumented_by_opentelemetry:
        existing.uninstrument()
    
    set_monocle_instrumentor(None)
    set_monocle_setup_signature(None)
    
    # Create test directory
    os.makedirs(TEST_TRACE_DIR, exist_ok=True)
    
    # Clean old test files
    for f in glob.glob(f"{TEST_TRACE_DIR}/monocle_trace_custom_span_*.json"):
        os.remove(f)
    
    # Setup telemetry with file exporter
    instrumentor = setup_monocle_telemetry(
        workflow_name="custom_span_demo",
        span_processors=[SimpleSpanProcessor(
            FileSpanExporter(out_path=TEST_TRACE_DIR)
        )]
    )
    
    yield
    
    # Cleanup
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, 'force_flush'):
        try:
            tracer_provider.force_flush()
        except Exception:
            pass
    
    if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    
    set_monocle_instrumentor(None)
    set_monocle_setup_signature(None)


def test_custom_span_creates_trace_file_with_io(setup):
    """
    Test that @monocle_trace_method creates trace files with input/output capture.
    
    This demonstrates what testers will see in production trace files:
    - Function inputs (args and kwargs) captured in data.input event
    - Function outputs (return values) captured in data.output event
    - Span type set to "custom"
    """
    
    # Define a realistic function with the decorator
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
    
    # Execute the function with test data
    test_items = [
        {"name": "Laptop", "price": 999.99, "quantity": 1},
        {"name": "Mouse", "price": 24.99, "quantity": 2},
        {"name": "Keyboard", "price": 79.99, "quantity": 1}
    ]
    
    result = calculate_order_total(test_items, tax_rate=0.08, discount_code="SAVE10")
    
    # Verify function executed correctly
    assert result is not None
    assert result["item_count"] == 3
    assert result["discount_applied"] == True
    
    # Force flush to write trace file
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, 'force_flush'):
        tracer_provider.force_flush()
    
    time.sleep(1.0)  # Wait for file write
    
    # Find trace file
    trace_files = glob.glob(f"{TEST_TRACE_DIR}/monocle_trace_custom_span_*.json")
    assert len(trace_files) > 0, f"No trace files found in {TEST_TRACE_DIR}"
    
    trace_file = trace_files[0]
    
    # Read and validate trace file
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)
    
    assert len(trace_data) > 0, "Trace file should contain spans"
    
    # Find custom span
    custom_span = next((s for s in trace_data if s.get("name") == "calculate_order_total"), None)
    assert custom_span is not None, "Custom span not found in trace"
    
    # Verify span attributes
    assert custom_span["attributes"]["span.type"] == "custom"
    assert custom_span["attributes"]["workflow.name"] == "custom_span_demo"
    
    # Verify input event
    events = custom_span.get("events", [])
    input_events = [e for e in events if e["name"] == "data.input"]
    assert len(input_events) == 1, "Should have data.input event"
    
    input_data = json.loads(input_events[0]["attributes"]["input"])
    assert len(input_data["args"]) == 1
    assert input_data["args"][0] == test_items
    assert input_data["kwargs"]["tax_rate"] == 0.08
    assert input_data["kwargs"]["discount_code"] == "SAVE10"
    
    # Verify output event
    output_events = [e for e in events if e["name"] == "data.output"]
    assert len(output_events) == 1, "Should have data.output event"
    
    output_data = json.loads(output_events[0]["attributes"]["response"])
    assert "result" in output_data
    assert output_data["result"]["item_count"] == 3
    assert output_data["result"]["discount_applied"] == True

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
