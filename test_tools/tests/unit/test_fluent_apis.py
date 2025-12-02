import pytest
import os
from monocle_test_tools import TraceAssertion
from span_loader import JSONSpanLoader

@pytest.fixture(scope="module")
def setup():
    """Fixture to create a TraceAssertion instance for testing."""
    validator = TraceAssertion()
    current_script_path = os.path.abspath(__file__)
    spans = JSONSpanLoader.from_json(os.path.join(os.path.dirname(current_script_path), "traces/trace1.json"))
    validator.memory_exporter.export(spans)
    yield validator
    validator.memory_exporter.clear()

def test_tool_invocation_span(setup):
    setup.called_tool("adk_book_hotel_5", "adk_hotel_booking_agent_5") \
        .has_input("{'city': 'Mumbai', 'hotel_name': 'Marriot Intercontinental'}") \
        .has_output("{'status': 'success', 'message': 'Successfully booked a stay at Marriot Intercontinental in Mumbai.'}") \
        .contains_input("Mumbai") \
        .contains_output("Successfully booked") \
        .does_not_contain_input("Delhi") \
        .does_not_contain_output("failed")

def test_agent_invocation(setup):
    setup.called_agent("adk_hotel_booking_agent_5") \
        .has_input("Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights.") \
        .contains_output("book a flight from San Francisco to Mumbai for 26th Nov 2025") \
        .does_not_have_output("cancel the booking") \
        .does_not_have_output("failed")

if __name__ == "__main__":
    pytest.main([__file__])