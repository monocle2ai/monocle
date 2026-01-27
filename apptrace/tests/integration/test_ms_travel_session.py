import logging
import pytest
import random
import time
import os
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

try:
    from agent_framework import ChatAgent
    from agent_framework.azure import AzureOpenAIAssistantsClient
    from azure.identity.aio import AzureCliCredential
    from typing import Annotated
    MICROSOFT_AGENT_AVAILABLE = True
except ImportError:
    MICROSOFT_AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Flight booking tool
def book_flight(
    from_airport: Annotated[str, "The departure airport code (e.g., JFK, LAX)"],
    to_airport: Annotated[str, "The destination airport code (e.g., SFO, ORD)"],
) -> str:
    """Book a flight from one airport to another"""
    confirmation = f"FL{random.randint(100000, 999999)}"
    cost = random.randint(300, 800)
    return f"FLIGHT BOOKING CONFIRMED #{confirmation}: {from_airport} to {to_airport} - ${cost}"


# Check for required environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") if MICROSOFT_AGENT_AVAILABLE else None
deployment = os.getenv("AZURE_OPENAI_API_DEPLOYMENT") if MICROSOFT_AGENT_AVAILABLE else None
api_key = os.getenv("AZURE_OPENAI_API_KEY") if MICROSOFT_AGENT_AVAILABLE else None


@pytest.fixture(scope="module")
def setup():
    """Setup telemetry instrumentation for Microsoft Agent Framework session tests."""
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter()
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(custom_exporter)
    ]
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="mic_ag_assistants_session_test",
            span_processors=span_processors,
        )
        yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.mark.skipif(not MICROSOFT_AGENT_AVAILABLE, reason="Microsoft Agent Framework not installed")
@pytest.mark.asyncio
async def test_assistants_multi_turn_session(setup):
    # Initialize Azure OpenAI Assistants client (server-managed threads)
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        client = AzureOpenAIAssistantsClient(
            endpoint=endpoint,
            deployment_name=deployment,
            api_version="2024-05-01-preview",
            api_key=api_key,
        )
    else:
        # Use Azure CLI authentication (requires: az login)
        client = AzureOpenAIAssistantsClient(
            endpoint=endpoint,
            deployment_name=deployment,
            api_version="2024-05-01-preview",
            credential=AzureCliCredential(),
        )
    
    # Create flight booking agent
    flight_agent = client.as_agent(
        name="MS_Flight_Booking_Agent",
        instructions=(
            "You are a Flight Booking Assistant. "
            "Your goal is to help users book flights between any two cities or airports. "
            "Book the requested flight and provide confirmation details."
        ),
        tools=[book_flight],
    )

    # Step 1: Let Azure create the thread (don't pass service_thread_id on first call)
    logger.info("Creating new Azure-managed thread...")
    thread = flight_agent.get_new_thread()
    
    # First interaction
    logger.info("First interaction: Book a flight from BOM to JFK for December 15th")
    response1 = await flight_agent.run("Book a flight from BOM to JFK for December 15th", thread=thread)
    logger.info(f"Agent Response 1: {response1.text}")
    
    assert response1, "Should get response from first interaction"
    assert response1.text, "Response should have text"
    
    # Thread ID is populated after the first run
    azure_thread_id = thread.service_thread_id
    logger.info(f"Azure Thread ID: {azure_thread_id}")
    assert azure_thread_id, "Thread should have an Azure service thread ID"
    assert azure_thread_id.startswith("thread_"), "Azure thread ID should start with 'thread_'"

    # Second interaction - continue with same thread
    logger.info("Second interaction: Book a return flight for December 20th")
    response2 = await flight_agent.run("Book a return flight for December 20th", thread=thread)
    logger.info(f"Agent Response 2: {response2.text}")
    
    logger.info("=" * 60)
    logger.info("Simulating session resume (like after app restart)")
    logger.info("=" * 60)
    
    # Step 3: Resume by passing Azure's thread_id as service_thread_id
    # Azure retrieves the stored conversation automatically
    resumed_thread = flight_agent.get_new_thread(service_thread_id=azure_thread_id)
    logger.info(f"Thread resumed with ID: {azure_thread_id}")
    assert resumed_thread.service_thread_id == azure_thread_id, "Resumed thread should have same ID"
    
    # Continue conversation - agent has full context from Azure-stored thread
    logger.info("Third interaction: What did we talk about?")
    response3 = await flight_agent.run("What did we talk about?", thread=resumed_thread)
    logger.info(f"Agent Response 3: {response3.text}")
    
    assert response3, "Should get response from resumed session"
    assert response3.text, "Response should have text"
    
    # Verify conversation context was maintained
    response_lower = response3.text.lower()
    assert any(keyword in response_lower for keyword in ["flight", "bom", "jfk", "december"]), \
        "Resumed conversation should reference previous flight bookings"
    
    logger.info("All conversation updates automatically saved to Azure")
    
    # Verify spans and session tracking
    verify_spans_with_session(setup, azure_thread_id)


def verify_spans_with_session(custom_exporter, expected_thread_id):
    """Verify spans contain scope.agentic.session attribute with the thread ID."""
    time.sleep(2)
    
    found_inference = False
    found_agent = False
    found_tool = False
    found_session_attribute = False
    session_ids_found = set()
    spans_with_session = []
    
    spans = custom_exporter.get_captured_spans()
    
    logger.info(f"Analyzing {len(spans)} spans for session tracking...")
    
    for span in spans:
        span_attributes = span.attributes
        
        # Check for session attribute
        if "scope.agentic.session" in span_attributes:
            found_session_attribute = True
            session_id = span_attributes["scope.agentic.session"]
            session_ids_found.add(session_id)
            spans_with_session.append({
                "span_name": span.name,
                "span_type": span_attributes.get("span.type"),
                "session_id": session_id
            })
            logger.info(f"Found session ID in span '{span.name}': {session_id}")

        # Check for inference spans
        if "span.type" in span_attributes and span_attributes["span.type"] in [
            "inference", "inference.framework", "inference.modelapi"
        ]:
            found_inference = True

        # Check for agent invocation spans
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.invocation"
            and "entity.1.name" in span_attributes
        ):
            assert span_attributes.get("entity.1.type") == "agent.microsoft", \
                "Agent entity should be of type agent.microsoft"
            assert span_attributes["entity.1.name"] == "MS_Flight_Booking_Agent", \
                "Agent name should be MS_Flight_Booking_Agent"
            found_agent = True

        # Check for tool invocation spans
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert span_attributes.get("entity.1.type") == "tool.microsoft", \
                "Tool entity should be of type tool.microsoft"
            assert span_attributes["entity.1.name"] == "book_flight", \
                "Tool name should be book_flight"
            found_tool = True

    # Assertions
    assert found_inference, "Inference span not found"
    assert found_agent, "Agent invocation span not found"
    assert found_tool, "Tool invocation span not found"
    
    # Critical assertion for session tracking
    assert found_session_attribute, \
        "scope.agentic.session attribute NOT FOUND in any spans!"
    
    logger.info(f"Session IDs found: {session_ids_found}")
    logger.info(f"Spans with session attribute: {len(spans_with_session)}")
    
    # Verify the session ID matches the Azure thread ID
    assert expected_thread_id in session_ids_found, \
        f"Expected thread ID '{expected_thread_id}' not found in session IDs: {session_ids_found}"
    
    logger.info(f"✓ Successfully verified scope.agentic.session attribute with thread ID: {expected_thread_id}")
    logger.info(f"✓ Found session tracking in {len(spans_with_session)} spans")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
