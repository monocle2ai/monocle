import pytest
from monocle_test_tools.pytest_plugin import monocle_trace_asserter
from test_common.adk_travel_agent import root_agent

@pytest.mark.asyncio
async def test_trace_level_sentiment_bias_evaluation(monocle_trace_asserter):
    """v0: Basic sentiment, bias evaluation on trace - only specify eval name and expected value."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")
    # Fact is implicit (trace), only specify eval template name and expected value
    monocle_trace_asserter.with_evaluation("okahu").check_eval("sentiment", "positive")\
        .check_eval("bias", "unbiased")

@pytest.mark.asyncio
async def test_trace_level_quality_metrics_evaluation(monocle_trace_asserter):
    """v0: Multiple evaluations on trace - frustration, hallucination, contextual_precision."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Please Book a flight from New York to Hamburg for 1st Dec 2025. Book a flight from Hamburg to Paris on January 1st. " \
                        "Then book a hotel room in Paris for 5th Jan 2026.")
    monocle_trace_asserter.with_evaluation("okahu").check_eval("frustration", "ok")\
        .check_eval("hallucination", "no_hallucination")
    monocle_trace_asserter.with_evaluation("okahu").check_eval("contextual_precision", "high_precision")

@pytest.mark.asyncio
async def test_filtered_agent_tool_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")
    
    monocle_trace_asserter.called_agent("adk_flight_booking_agent_5")
    monocle_trace_asserter.with_evaluation("okahu").check_eval("conversation_completeness", "complete")

    monocle_trace_asserter.called_tool("adk_book_flight_5","adk_flight_booking_agent_5")
    monocle_trace_asserter.with_evaluation("okahu").check_eval("toxicity", "non_toxic")\
        .check_eval("contextual_relevancy", "highly_relevant")
@pytest.mark.asyncio
async def test_complex_workflow_summarization_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", 
                        "Book a flight from San Francisco to Mumbai for 26th April 2026. Book a two queen room at Marriott Intercontinental at Central Mumbai for 27th April 2026 for 4 nights.")
		
    monocle_trace_asserter.with_evaluation("okahu").check_eval("summarization", "excellent")

if __name__ == "__main__":
    pytest.main([__file__]) 