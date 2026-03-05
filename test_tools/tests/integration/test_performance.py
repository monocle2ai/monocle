import pytest
from monocle_test_tools.pytest_plugin import monocle_trace_asserter
from test_common.adk_travel_agent import root_agent

# test goals in order:
# 1) Test basic functionality of performance assertions (token and duration limits) on the entire workflow trace
# 2) Test performance assertions on filtered spans (specific agents/tools)
# 3) Test chained performance assertions (token and duration limits together)
# 4) Test performance assertions on a more complex workflow with multiple agents/tools and longer duration
@pytest.mark.asyncio
async def test_trace_level_limits(monocle_trace_asserter):
    """Test basic token and duration limits on full workflow trace."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")
    # verify token limit is under 1050 and duration under 12.5 seconds for the entire workflow
    monocle_trace_asserter.under_token_limit(1050)
    monocle_trace_asserter.under_duration(12.5)


@pytest.mark.asyncio
async def test_filtered_span_limits(monocle_trace_asserter):
    """Test performance assertions on filtered spans (agent/tool specific)."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Please Book a flight from New York to Hamburg for 1st Dec 2025. Book a flight from Hamburg to Paris on January 1st. " \
                        "Then book a hotel room in Paris for 5th Jan 2026.")

    # verify token limit is under 1500 and duration under 10 seconds for the flight booking agent/tool spans
    # This will fail as the necessary spans for duration calculation are excluded by the filter.
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5").under_duration(10)
    monocle_trace_asserter.called_tool("adk_book_flight_5").under_token_limit(100)


@pytest.mark.asyncio
async def test_chained_filtered_limits(monocle_trace_asserter):
    """Test chained performance assertions after agent filtering."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")    
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5")

    # verify token limit is under 1070 and duration under 10 seconds for the entire workflow
    monocle_trace_asserter.under_token_limit(1070).under_duration(10)

@pytest.mark.asyncio
async def test_complex_workflow_limits(monocle_trace_asserter):
    """Test performance limits on multi-agent workflow with longer duration."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        "Book a flight from San Francisco to Mumbai for 26th April 2026. Book a two queen room at Marriott Intercontinental at Central Mumbai for 27th April 2026 for 4 nights.")

    # verify token limit is under 2000 and duration under 15 seconds for the entire workflow
    monocle_trace_asserter.under_token_limit(2000)
    monocle_trace_asserter.under_duration(10)

if __name__ == "__main__":
    pytest.main([__file__]) 