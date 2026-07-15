import pytest
import threading
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.exporters.file_exporter import FileSpanExporter


@pytest.fixture(scope="function")
def in_memory_exporter():
    """Fixture to provide an in-memory span exporter for testing."""
    exporter = InMemorySpanExporter()
    return exporter


def test_workflow_name_with_external_instrumentation(in_memory_exporter):
    """
    Test that workflow_name correctly uses SERVICE_NAME from external instrumentation
    when external instrumentation has already initialized the TracerProvider.
    
    This simulates scenarios where cloud platforms (Azure, AWS, GCP) or APM tools
    set up OpenTelemetry instrumentation before Monocle is initialized.
    """
    
    
    # Step 1: Simulate external/automatic instrumentation setting service.name FIRST
    external_service_name = "my-application-service"
    external_resource = Resource(attributes={
        SERVICE_NAME: external_service_name,
        "deployment.environment": "production",
        "service.version": "1.0.0",
    })
    
    provider = TracerProvider(resource=external_resource)
    trace.set_tracer_provider(provider)
    
    # Step 2: Application initializes Monocle with workflow_name
    workflow_name = "my_ai_workflow"
    
    span_processors = [
        SimpleSpanProcessor(in_memory_exporter),
        BatchSpanProcessor(FileSpanExporter()),
    ]
    
    instrumentor = setup_monocle_telemetry(
        workflow_name=workflow_name,
        span_processors=span_processors,
    )
    
    # Step 3: Make OpenAI API call in a worker thread
    def worker_thread_task():
        from openai import OpenAI
        client = OpenAI()
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "Say hello in one word"}
                ],
                max_tokens=5
            )
        except Exception as e:
            # API call may fail in test environment, but spans should still be created
            pass
    
    # Create and start worker thread
    worker_thread = threading.Thread(target=worker_thread_task)
    worker_thread.start()
    
    # Wait for worker thread to complete
    worker_thread.join()
    
    # Step 4: Verify the workflow name is set correctly in spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) >= 1, "Should have at least one span from OpenAI instrumentation"
    
    # Find the OpenAI span (should contain "openai" or "chat.completions" in the name)
    openai_span = None
    for span in spans:
        if "openai" in span.name.lower() or "chat" in span.name.lower():
            openai_span = span
            break
    
    assert openai_span is not None, "Should have an OpenAI-related span"
    
    # Assert workflow.name should be from monocle instrumentation
    assert "workflow.name" in openai_span.attributes, "workflow.name attribute should be present"
    assert openai_span.attributes["workflow.name"] == workflow_name, \
        f"Expected workflow.name to be '{workflow_name}' (from monocle instrumentation), got '{openai_span.attributes.get('workflow.name')}'"
    
    resource_attributes = dict(openai_span.resource.attributes)
    assert resource_attributes.get(SERVICE_NAME) == external_service_name, \
        f"Expected service.name to be '{external_service_name}' (from Monocle), got '{resource_attributes.get(SERVICE_NAME)}'"
    
    # Cleanup
    in_memory_exporter.clear()
    if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    trace._TRACER_PROVIDER = None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])