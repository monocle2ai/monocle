import pytest
from test_common.adk_travel_agent import root_agent

# This test is to check that the agent calls are being captured correctly and the evaluation is working as expected.
@pytest.mark.asyncio
async def test_agent_invocation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5").contains_input("Book a flight from San Jose to Seattle")
    monocle_trace_asserter.with_evaluation("okahu").check_eval("sentiment", "positive")

# This test is to check that the tool calls are being captured correctly and the evaluation is working as expected.
@pytest.mark.asyncio
async def test_tool_invocation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        "Book a flight from San Francisco to Mumbai for 26th Nov 2025")
    monocle_trace_asserter.called_tool("adk_book_flight_5","adk_flight_booking_agent_5").contains_input("Mumbai")
    monocle_trace_asserter.with_evaluation("okahu").check_eval("sentiment", "positive")

# This test is similar to the above test but with more complex input and multiple tool calls. 
# This is to test that the asserter can correctly identify and assert on multiple tool calls in a single trace.
@pytest.mark.asyncio
async def test_tool_invocation1(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        "Book a flight from San Francisco to Mumbai for 26th April 2026. Book a two queen room at Marriott Intercontinental at Central Mumbai for 27th April 2026 for 4 nights.")
    
    monocle_trace_asserter.called_tool("adk_book_flight","adk_flight_booking_agent") \
        .contains_input("Mumbai").contains_input("San Francisco").contains_input("26th April 2026") \
        .contains_output("San Francisco to Mumbai").contains_output("success")
    
    monocle_trace_asserter.called_tool("adk_book_hotel","adk_hotel_booking_agent") \
        .contains_input("Central Mumbai").contains_input("27th April 2026").contains_input("Marriott Intercontinental") \
        .contains_output("booked") \
        .contains_output("Successfully booked a stay at Marriott Intercontinental in Central Mumbai") \
        .contains_output("success")
		
    monocle_trace_asserter.with_evaluation("okahu").check_eval("sentiment", "positive")

if __name__ == "__main__":
    pytest.main([__file__]) 