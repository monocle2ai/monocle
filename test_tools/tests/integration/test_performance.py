import pytest
from monocle_test_tools.pytest_plugin import monocle_trace_asserter
from test_common.adk_travel_agent import root_agent

# test goals in order:
# 1) Test basic functionality of performance assertions (token and duration limits) on the entire workflow trace with multiple fact names
# 2) Test performance assertions on filtered spans (specific agents/tools) with agent_invocation and tool_invocation fact names
# 3) Test chained performance assertions (token and duration limits together) with agent_turn and inference fact names
# 4) Test performance assertions on a more complex workflow with multiple fact names (workflow, agent_invocation, tool_invocation)
@pytest.mark.asyncio
async def test_trace_level_limits(monocle_trace_asserter):
    """Test basic token and duration limits on full workflow trace."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")
    # verify total tokens are under 1100 and each individual workflow span is under 12.5 seconds
    monocle_trace_asserter.under_token_limit(1100)
    monocle_trace_asserter.under_duration(12.5, units="seconds", fact_name="workflow")
    # also verify each inference span is under 5000 milliseconds
    monocle_trace_asserter.under_duration(5000, units="ms", fact_name="inference")


@pytest.mark.asyncio
async def test_filtered_span_limits(monocle_trace_asserter):
    """Test performance assertions on filtered spans (agent/tool specific)."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Please Book a flight from New York to Hamburg for 1st Dec 2025. Book a flight from Hamburg to Paris on January 1st. " \
                        "Then book a hotel room in Paris for 5th Jan 2026.")

    # verify each individual agent invocation span is under 0.2 minutes
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5").under_duration(0.2, units="minutes", fact_name="agent_invocation")
    # verify each tool invocation span is under 1 millisecond and uses under 100 tokens
    monocle_trace_asserter.called_tool("adk_book_flight_5").under_duration(1, units="ms", fact_name="tool_invocation").under_token_limit(100)


@pytest.mark.asyncio
async def test_chained_filtered_limits(monocle_trace_asserter):
    """Test chained performance assertions after agent filtering."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")    
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5").under_duration(0.2, units="minutes", fact_name="agent_invocation")
    monocle_trace_asserter.called_tool("adk_book_flight_5").under_duration(4000, units="ms", fact_name="tool_invocation").under_token_limit(1070)

    # verify total tokens are under 1070, each agent turn is under 0.2 minutes, and each inference is under 4000 ms
    monocle_trace_asserter.under_token_limit(1070).under_duration(0.2, units="minutes", fact_name="agent_turn").under_duration(4000, units="ms", fact_name="inference")

@pytest.mark.asyncio
async def test_complex_workflow_limits(monocle_trace_asserter):
    """Test performance limits on multi-agent workflow with longer duration."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        "Book a flight from San Francisco to Mumbai for 26th April 2026. Book a two queen room at Marriott Intercontinental at Central Mumbai for 27th April 2026 for 4 nights.")

    # verify total tokens are under 2000
    monocle_trace_asserter.under_token_limit(2000)
    # verify each workflow span is under 10 seconds, each agent invocation is under 0.2 minutes,
    # and each tool invocation is under 6000 milliseconds
    monocle_trace_asserter.under_duration(10, units="seconds", fact_name="workflow")
    monocle_trace_asserter.under_duration(0.2, units="minutes", fact_name="agent_invocation")
    monocle_trace_asserter.under_duration(6000, units="ms", fact_name="tool_invocation")


@pytest.mark.asyncio
async def test_multiple_filtered_assertions(monocle_trace_asserter):
    """Test multiple filtered assertions using called_agent and called_tool on same asserter instance."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Chicago to Tokyo for 15th August 2026. Then book a hotel room in Tokyo for 16th August 2026.")
    
    # verify specific flight booking agent invocations are under duration limit
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5").under_duration(0.15, units="minutes", fact_name="agent_invocation")
    # verify specific hotel booking agent invocations with multiple constraints
    monocle_trace_asserter.called_agent("adk_hotel_booking_agent_5").under_duration(0.18, units="minutes", fact_name="agent_invocation").under_token_limit(900)
    # verify book flight tool invocations are under duration and token limits
    monocle_trace_asserter.called_tool("adk_book_flight_5").under_duration(5000, units="ms", fact_name="tool_invocation").under_token_limit(80)
    # verify book hotel tool invocations are under duration limit
    monocle_trace_asserter.called_tool("adk_book_hotel_5").under_duration(7000, units="ms", fact_name="tool_invocation")
    # verify agent turns for flight booking agent are under duration limit and token limit
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5").called_tool("adk_book_flight_5").under_duration(0.1, units="minutes", fact_name="tool_invocation")


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Workflow duration is expected to exceed the low limit set in this test.")
async def test_workflow_duration_failure(monocle_trace_asserter):
    """Test that workflow duration limit failure is detected - this test is expected to fail."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")
    # Set unrealistically low duration limit in milliseconds - should fail
    monocle_trace_asserter.under_duration(1, units="ms", fact_name="workflow")


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Inference duration is expected to exceed the low limit set in this test.")
async def test_inference_duration_failure(monocle_trace_asserter):
    """Test that inference duration limit failure is detected - this test is expected to fail."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from New York to Paris for 15th March 2026.")
    # Set unrealistically low duration limit for inference spans in seconds - should fail
    monocle_trace_asserter.under_duration(0.01, units="seconds", fact_name="inference")


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Agent invocation duration is expected to exceed the low limit set in this test.")
async def test_agent_invocation_duration_failure(monocle_trace_asserter):
    """Test that agent invocation duration limit failure is detected."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a hotel room in Paris for 5th January 2026.")
    # Set unrealistically low duration limit for agent invocation spans in minutes - should fail
    monocle_trace_asserter.under_duration(0.0001, units="minutes", fact_name="agent_invocation")


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Tool invocation duration is expected to exceed the low limit set in this test.")
async def test_tool_invocation_duration_failure(monocle_trace_asserter):
    """Test that tool invocation duration limit failure is detected."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Boston to London for 20th May 2026.")
    # Set unrealistically low duration limit for tool invocation spans in milliseconds - should fail
    monocle_trace_asserter.under_duration(0.1, units="ms", fact_name="tool_invocation")


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Agent turn duration is expected to exceed the low limit set in this test.")
async def test_agent_turn_duration_failure(monocle_trace_asserter):
    """Test that agent turn duration limit failure is detected."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Seattle to Tokyo for 10th June 2026.")
    # Set unrealistically low duration limit for agent turn spans in seconds - should fail
    monocle_trace_asserter.under_duration(0.1, units="seconds", fact_name="agent_turn")


if __name__ == "__main__":
    pytest.main([__file__]) 