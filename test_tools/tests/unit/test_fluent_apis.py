import pytest
import os
from monocle_test_tools import TraceAssertion
from span_loader import JSONSpanLoader
os.environ["MONOCLE_EXPORT_FAILED_TESTS_ONLY"] = "true"

def test_tool_invocation_span(monocle_trace_asserter:TraceAssertion):
    monocle_trace_asserter.load_spans(JSONSpanLoader.load_spans("traces/trace1.json"))
    monocle_trace_asserter.called_tool("adk_book_hotel_5", "adk_hotel_booking_agent_5") \
        .has_input("{'city': 'Mumbai', 'hotel_name': 'Marriot Intercontinental'}") \
        .has_output("{'status': 'success', 'message': 'Successfully booked a stay at Marriot Intercontinental in Mumbai.'}") \
        .contains_input("Mumbai") \
        .contains_output("Successfully booked") \
        .does_not_contain_input("Delhi") \
        .does_not_contain_output("failed")

def test_agent_invocation(monocle_trace_asserter:TraceAssertion):
    monocle_trace_asserter.load_spans(JSONSpanLoader.load_spans("traces/trace1.json"))
    monocle_trace_asserter.called_agent("adk_hotel_booking_agent_5") \
        .has_input("Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights.") \
        .contains_output("I have booked a stay at Marriot Intercontinental in Mumbai.") \
        .does_not_have_output("cancel the booking") \
        .does_not_have_output("failed")

if __name__ == "__main__":
    pytest.main([__file__])
