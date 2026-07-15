import logging
import pytest
import time
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

# Import CrewAI components
try:
    from crewai import Agent, Crew, Task
    from crewai.tools import BaseTool
    from langchain_openai import ChatOpenAI
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

logger = logging.getLogger(__name__)


# Simple tool - all verification happens in backend automatically
class BookHotelTool(BaseTool):
    name: str = "book_hotel"
    description: str = "Book a hotel reservation"

    def _run(self, hotel_name: str, city: str, nights: int = 1) -> str:
        import time
        import random
        
        # Simple booking simulation
        time.sleep(0.1)
        confirmation = f"BK{random.randint(100000, 999999)}"
        cost = nights * 150
        
        return f"YOUR BOOKING CONFIRMED #{confirmation}: {hotel_name} in {city} for {nights} nights - ${cost}"


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for CrewAI tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter)
    ]

    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="crewai_simple_test",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def create_simple_crew():
    """Create a simple CrewAI crew with 1 agent and 1 tool."""
    if not CREWAI_AVAILABLE:
        pytest.skip("CrewAI not available")

    # Create simple tool
    hotel_tool = BookHotelTool()

    # Create simple agent
    agent = Agent(
        role="Hotel Booking Agent",
        goal="Book hotels for travelers",
        backstory="You help people book hotels.",
        tools=[hotel_tool],
        llm=ChatOpenAI(model="gpt-4.1"),
        verbose=True
    )

    # Create simple task
    task = Task(
        name = "Hotel Booking Task",
        description="Book a hotel room at Marriott in New York for 2 nights.",
        expected_output="Hotel booking confirmation",
        agent=agent
    )

    # Create simple crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    return crew


@pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not installed")
def test_crewai_simple(setup):
    """Simple CrewAI test with 1 agent and 1 tool."""
    crew = create_simple_crew()
    
    # Execute the crew with simple input
    result = crew.kickoff(inputs={
        "hotel_request": "Book Marriott Hotel in New York for 2 nights for next week"
    })
    logger.info(f"Result: {result}")
    
    # Basic verification
    assert result is not None, "Should get a result"
    verify_spans(setup)


def verify_spans(custom_exporter):
    time.sleep(2)
    found_inference = found_agent = found_tool = found_tool_call = False
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
                span_attributes["span.type"] == "inference"
                or span_attributes["span.type"] == "inference.framework"
        ):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4.1"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4.1"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            assert "finish_reason" in span_metadata.attributes
            if span_metadata.attributes["finish_reason"] == "tool_calls":
                found_tool_call =True
            found_inference = True

        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.invocation"
                and "entity.1.name" in span_attributes
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.crewai"
            if span_attributes["entity.1.name"] == "Hotel Booking Agent":
                found_agent = True
        if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.crewai"
            if span_attributes["entity.1.name"] == "book_hotel":
                found_tool = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_tool_call, "Tool call finish reason not found"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
