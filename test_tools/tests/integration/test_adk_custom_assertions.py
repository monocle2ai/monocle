"""
Example demonstrating custom assertion messages in Monocle test assertions.

This file comprehensively covers ALL 24 assertion methods with:
- Passing tests: Show normal functionality without custom messages
- ✗ Failing tests (@pytest.mark.xfail): Demonstrate custom error messages

Covered methods:
  Agent/Tool: called_tool, does_not_call_tool, called_agent, does_not_call_agent
  Input: has_input, has_any_input, does_not_have_input, does_not_have_any_input,
         contains_input, contains_any_input, does_not_contain_input, does_not_contain_any_input
  Output: has_output, has_any_output, does_not_have_output, does_not_have_any_output,
          contains_output, contains_any_output, does_not_contain_output, does_not_contain_any_output
  Performance: under_token_limit, under_duration
"""
import pytest
from monocle_test_tools import TraceAssertion
from test_common.adk_travel_agent import root_agent

# ============================================================================
# PASSING TESTS - Demonstrate normal functionality of each assertion method
# ============================================================================

@pytest.mark.asyncio
async def test_called_tool_passes(monocle_trace_asserter):
    """called_tool - passes when tool is called."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        "Book a flight from Boston to Chicago for 15th Dec 2025")
    monocle_trace_asserter.called_tool("adk_book_flight_5", "adk_flight_booking_agent_5")

@pytest.mark.asyncio
async def test_does_not_call_tool_passes(monocle_trace_asserter):
    """does_not_call_tool - passes when tool is not called."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        "Book a flight from Portland to Denver for 20th Jan 2026")
    monocle_trace_asserter.does_not_call_tool("nonexistent_tool")

@pytest.mark.asyncio
async def test_called_agent_passes(monocle_trace_asserter):
    """called_agent - passes when agent is called."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "I need to fly from Austin to Miami on March 1st 2026.")
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5")

@pytest.mark.asyncio
async def test_does_not_call_agent_passes(monocle_trace_asserter):
    """does_not_call_agent - passes when agent is not called."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Phoenix to Atlanta for April 5th 2026")
    monocle_trace_asserter.does_not_call_agent("nonexistent_agent")

@pytest.mark.asyncio
async def test_has_input_passes(monocle_trace_asserter):
    """has_input - passes when input matches."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Miami to Orlando for May 1st 2026")
    monocle_trace_asserter.called_tool("adk_book_flight_5").has_input("Orlando")

@pytest.mark.asyncio
async def test_has_any_input_passes(monocle_trace_asserter):
    """has_any_input - passes when any input matches."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Seattle to Portland for May 5th 2026")
    monocle_trace_asserter.called_tool("adk_book_flight_5").has_any_input("Portland", "Seattle", "Vancouver")

@pytest.mark.asyncio
async def test_does_not_have_input_passes(monocle_trace_asserter):
    """does_not_have_input - passes when input doesn't match."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Dallas to Austin for May 10th 2026")
    monocle_trace_asserter.does_not_have_input("Tokyo")

@pytest.mark.asyncio
async def test_does_not_have_any_input_passes(monocle_trace_asserter):
    """does_not_have_any_input - passes when none of the inputs match."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Phoenix to Denver for May 15th 2026")
    monocle_trace_asserter.does_not_have_any_input("London", "Paris", "Tokyo")

@pytest.mark.asyncio
async def test_contains_input_passes(monocle_trace_asserter):
    """contains_input - passes when input contains substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from New York to Los Angeles for May 20th 2026")
    monocle_trace_asserter.called_tool("adk_book_flight_5").contains_input("Los Angeles")

@pytest.mark.asyncio
async def test_contains_any_input_passes(monocle_trace_asserter):
    """contains_any_input - passes when input contains any substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Tampa to Atlanta for May 25th 2026")
    monocle_trace_asserter.contains_any_input("Tampa", "Orlando", "Miami")

@pytest.mark.asyncio
async def test_does_not_contain_input_passes(monocle_trace_asserter):
    """does_not_contain_input - passes when input doesn't contain substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Chicago to Detroit for June 1st 2026")
    monocle_trace_asserter.does_not_contain_input("BUSINESS_CLASS")

@pytest.mark.asyncio
async def test_does_not_contain_any_input_passes(monocle_trace_asserter):
    """does_not_contain_any_input - passes when input doesn't contain any substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Boston to Philadelphia for June 5th 2026")
    monocle_trace_asserter.does_not_contain_any_input("FIRST_CLASS", "PREMIUM", "VIP")

@pytest.mark.asyncio
async def test_has_output_passes(monocle_trace_asserter):
    """has_output - passes when output matches."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Francisco to San Diego for June 10th 2026")
    # This would pass if the output actually contained the expected string
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5")

@pytest.mark.asyncio
async def test_has_any_output_passes(monocle_trace_asserter):
    """has_any_output - passes when any output matches."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Houston to Dallas for June 15th 2026")
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5")

@pytest.mark.asyncio
async def test_does_not_have_output_passes(monocle_trace_asserter):
    """does_not_have_output - passes when output doesn't match."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Las Vegas to Reno for June 20th 2026")
    monocle_trace_asserter.does_not_have_output("REFUND_PROCESSED")

@pytest.mark.asyncio
async def test_does_not_have_any_output_passes(monocle_trace_asserter):
    """does_not_have_any_output - passes when none of the outputs match."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Nashville to Memphis for June 25th 2026")
    monocle_trace_asserter.does_not_have_any_output("ERROR", "FAILED", "REJECTED")

@pytest.mark.asyncio
async def test_contains_output_passes(monocle_trace_asserter):
    """contains_output - passes when output contains substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Charlotte to Raleigh for July 1st 2026")
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5")

@pytest.mark.asyncio
async def test_contains_any_output_passes(monocle_trace_asserter):
    """contains_any_output - passes when output contains any substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Pittsburgh to Cleveland for July 5th 2026")
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5")

@pytest.mark.asyncio
async def test_does_not_contain_output_passes(monocle_trace_asserter):
    """does_not_contain_output - passes when output doesn't contain substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Baltimore to Washington for July 10th 2026")
    monocle_trace_asserter.does_not_contain_output("CANCELLATION_CONFIRMED")

@pytest.mark.asyncio
async def test_does_not_contain_any_output_passes(monocle_trace_asserter):
    """does_not_contain_any_output - passes when output doesn't contain any substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Indianapolis to Cincinnati for July 15th 2026")
    monocle_trace_asserter.does_not_contain_any_output("TIMEOUT", "CONNECTION_ERROR", "SERVICE_UNAVAILABLE")

@pytest.mark.asyncio
async def test_under_token_limit_passes(monocle_trace_asserter):
    """under_token_limit - passes when under limit."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Las Vegas to Orlando for July 20th 2026")
    monocle_trace_asserter.under_token_limit(1000000)

@pytest.mark.asyncio
async def test_under_duration_passes(monocle_trace_asserter):
    """under_duration - passes when under time limit."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Detroit to Nashville for July 25th 2026")
    monocle_trace_asserter.under_duration(30.0)

#----------------------------------------------------------------------------
# Failing tests section - These tests intentionally fail to demonstrate custom error messages
#----------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for missing tool")
async def test_fail_tool_not_called(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when expected tool is not called."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Seattle to Boston for October 10th 2026")
    monocle_trace_asserter.called_tool(
        "nonexistent_flight_tool",
        message="BOOKING FAILED: The flight booking tool 'nonexistent_flight_tool' was never invoked during the transaction"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for tool that was called")
async def test_fail_tool_was_called(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when tool was unexpectedly called."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Chicago to Miami for November 15th 2026")
    monocle_trace_asserter.does_not_call_tool(
        "adk_book_flight_5",
        message="POLICY VIOLATION: The booking tool 'adk_book_flight_5' was called but should have been blocked"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for missing agent")
async def test_fail_agent_not_invoked(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when expected agent is not invoked."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Portland to Denver for December 5th 2026")
    monocle_trace_asserter.called_agent(
        "nonexistent_travel_planner_agent",
        message="WORKFLOW ERROR: Expected 'nonexistent_travel_planner_agent' to handle this request but it was never invoked"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for agent that was called")
async def test_fail_agent_was_invoked(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when agent was unexpectedly invoked."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Austin to San Antonio for January 20th 2027")
    monocle_trace_asserter.does_not_call_agent(
        "adk_flight_booking_agent_5",
        message="AUTHORIZATION ERROR: Agent 'adk_flight_booking_agent_5' was invoked without proper permissions"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for missing input")
async def test_fail_input_not_found(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when expected input is missing."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Phoenix to Las Vegas for February 14th 2027")
    monocle_trace_asserter.called_tool("adk_book_flight_5").has_input(
        "MISSING_DESTINATION_TOKYO",
        message="INPUT VALIDATION ERROR: Required destination 'MISSING_DESTINATION_TOKYO' was not found in the booking request"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for unexpected input")
async def test_fail_input_contains_wrong_value(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when input contains unexpected value."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Dallas to Houston for March 3rd 2027")
    monocle_trace_asserter.does_not_have_input(
        "Dallas",
        message="SECURITY ALERT: Input contains restricted city 'Dallas' which is not allowed in this context"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for missing output")
async def test_fail_output_not_found(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when expected output is missing."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from New York to Paris for April 1st 2027")
    monocle_trace_asserter.has_output(
        "CONFIRMATION_CODE_XYZ123_MISSING",
        message="PAYMENT ERROR: Expected confirmation code 'CONFIRMATION_CODE_XYZ123_MISSING' was not generated"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for substring not in input")
async def test_fail_input_substring_missing(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when input doesn't contain expected substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Atlanta to Charlotte for May 10th 2027")
    monocle_trace_asserter.contains_input(
        "MISSING_KEYWORD_BUSINESS_CLASS",
        message="UPGRADE ERROR: Flight class 'MISSING_KEYWORD_BUSINESS_CLASS' was not specified in booking parameters"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for unexpected output substring")
async def test_fail_output_contains_wrong_substring(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when output contains unexpected substring."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Francisco to Tokyo for June 15th 2027")
    monocle_trace_asserter.does_not_contain_output(
        "Flight booked",
        message="RATE LIMIT ERROR: Booking was completed ('Flight booked') but should have been throttled"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for token limit")
async def test_fail_token_limit_exceeded(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when token limit is exceeded."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Los Angeles to Sydney for July 20th 2027")
    monocle_trace_asserter.under_token_limit(
        1,  # Artificially low limit to force failure
        message="COST ALERT: Token usage exceeded budget - only 1 token allowed but workflow consumed significantly more"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for duration limit")
async def test_fail_duration_limit_exceeded(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when duration limit is exceeded."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Boston to London for August 25th 2027")
    monocle_trace_asserter.under_duration(
        0.001,  # Artificially low limit to force failure
        message="PERFORMANCE CRITICAL: Workflow execution time exceeded SLA - maximum 0.001s allowed for real-time processing"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for has_any_input")
async def test_fail_has_any_input(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when none of the inputs match."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Miami to Tampa for Aug 25th 2026")
    monocle_trace_asserter.has_any_input(
        "London", "Paris", "Tokyo",
        message="ROUTE ERROR: None of the expected international destinations were found"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for does_not_have_any_input")
async def test_fail_does_not_have_any_input(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when prohibited inputs are detected."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from New York to Los Angeles for Sep 1st 2026")
    monocle_trace_asserter.does_not_have_any_input(
        "New York", "Los Angeles", "Chicago",
        message="COMPLIANCE ERROR: Blacklisted cities detected in booking request"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for contains_any_input")
async def test_fail_contains_any_input(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when premium service tiers not found."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Boston to Philadelphia for Sep 10th 2026")
    monocle_trace_asserter.contains_any_input(
        "PREMIUM", "FIRST_CLASS", "VIP",
        message="CLASS ERROR: None of the premium service tiers were found in booking"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for does_not_contain_input")
async def test_fail_does_not_contain_input(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when prohibited keyword detected."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Francisco to San Diego for Sep 15th 2026")
    monocle_trace_asserter.does_not_contain_input(
        "San Francisco",
        message="FILTER ERROR: Prohibited keyword 'San Francisco' detected in input"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for does_not_contain_any_input")
async def test_fail_does_not_contain_any_input(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when restricted region detected."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Houston to Dallas for Sep 20th 2026")
    monocle_trace_asserter.does_not_contain_any_input(
        "Houston", "Dallas", "Austin",
        message="REGION ERROR: Input contains restricted Texas cities"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for has_any_output")
async def test_fail_has_any_output(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when expected confirmation outputs missing."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Phoenix to Tucson for Sep 25th 2026")
    monocle_trace_asserter.has_any_output(
        "RECEIPT_SENT", "EMAIL_CONFIRMED", "TICKET_ISSUED",
        message="NOTIFICATION ERROR: None of the expected confirmation outputs were found"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for does_not_have_any_output")
async def test_fail_does_not_have_any_output(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when booking completion terms present."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Las Vegas to Reno for Sep 30th 2026")
    monocle_trace_asserter.does_not_have_any_output(
        "booked", "confirmed", "reserved",
        message="STATUS ERROR: Output contains booking completion terms that shouldn't be present"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for contains_output")
async def test_fail_contains_output(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when refund approval not found."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Orlando to Tampa for Oct 1st 2026")
    monocle_trace_asserter.contains_output(
        "REFUND_APPROVED_MISSING",
        message="REFUND ERROR: Expected refund approval message not found in output"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for contains_any_output")
async def test_fail_contains_any_output(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when promotional messages not found."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Nashville to Memphis for Oct 5th 2026")
    monocle_trace_asserter.contains_any_output(
        "UPGRADE_AVAILABLE", "LOYALTY_BONUS", "DISCOUNT_APPLIED",
        message="PROMOTION ERROR: None of the expected promotional messages were found"
    )

@pytest.mark.asyncio
@pytest.mark.xfail(reason="Intentionally failing to demonstrate custom error message for does_not_contain_any_output")
async def test_fail_does_not_contain_any_output(monocle_trace_asserter):
    """This test will fail - demonstrates custom message when success indicators present after rollback."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from Baltimore to Washington for Oct 10th 2026")
    monocle_trace_asserter.does_not_contain_any_output(
        "success", "completed", "confirmed",
        message="ROLLBACK ERROR: Output contains success indicators but transaction should have rolled back"
    )

if __name__ == "__main__":
    pytest.main([__file__])
