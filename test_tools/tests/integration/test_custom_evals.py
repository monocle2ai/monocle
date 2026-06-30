import pytest
from pathlib import Path
from monocle_test_tools.pytest_plugin import monocle_trace_asserter
from test_common.adk_travel_agent import root_agent

# All four tests share this custom template (classifies user input as
# valid / ambiguous / invalid relative to the agent's inferred purpose).
TEMPLATE_PATH = str(Path(__file__).parent / "custom_templates" / "user_input_validity.json")


@pytest.mark.asyncio
async def test_custom_template_traces(monocle_trace_asserter):
    """Custom user_input_validity template, traces fact — valid booking request."""
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk",
        "Book a flight from San Jose to Seattle for 27th Nov 2025.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="traces",
        expected="valid",
    )


@pytest.mark.asyncio
async def test_custom_template_inferences(monocle_trace_asserter):
    """Custom user_input_validity template, inferences fact — valid booking request."""
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk",
        "Book a flight from Boston to Miami for 15th Feb 2026.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="inferences",
        expected="valid",
    )


@pytest.mark.asyncio
async def test_custom_template_agentic_turns(monocle_trace_asserter):
    """Custom user_input_validity template, agentic_turns fact — valid booking request."""
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk",
        "Book a flight from Seattle to Portland for 10th April 2026.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="agentic_turns",
        expected="valid",
    )


@pytest.mark.asyncio
async def test_custom_template_agentic_sessions(monocle_trace_asserter):
    """Custom user_input_validity template, agentic_sessions fact — valid booking request."""
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk",
        "Book a flight from Dallas to Houston for 1st May 2026 and book a hotel for 2 nights.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="agentic_sessions",
        expected="valid",
    )


if __name__ == "__main__":
    pytest.main([__file__])
