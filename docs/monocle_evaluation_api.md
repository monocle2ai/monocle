# Monocle Evaluation API Usage

This document explains how to use the Monocle evaluation testing utilities with the `monocle_trace_asserter` fixture for validating AI agent quality, safety, and performance.

## Prerequisites

```bash
pip install monocle_test_tools
```

**Required Environment Variables:**
- `OKAHU_API_KEY` - Must be set to run evaluations
- `MONOCLE_EXPORTER` - Must be set to `okahu` or `file, okahu`

The evaluation tool will not run without these environment variables configured.

## 1. `monocle_trace_asserter` (Pytest Fixture)
A pytest fixture for running agents and asserting on evaluation metrics across entire traces or filtered spans.

**Example (Trace-level evaluation):**
```python
import pytest
from monocle_test_tools.pytest_plugin import monocle_trace_asserter

@pytest.mark.asyncio
async def test_sentiment_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk",
        "Book a flight from San Jose to Seattle for 27th Nov 2025."
    )
    
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("sentiment", "positive")\
        .check_eval("bias", "unbiased")
```

**Example (Filtered agent evaluation):**
```python
@pytest.mark.asyncio
async def test_agent_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    monocle_trace_asserter.called_agent("flight_booking_agent")
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("conversation_completeness", "complete")
```

**Example (Filtered tool evaluation):**
```python
@pytest.mark.asyncio
async def test_tool_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    monocle_trace_asserter.called_tool("book_flight", "flight_booking_agent")
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("toxicity", "non_toxic")\
        .check_eval("contextual_relevancy", "highly_relevant")
```

## 2. Available Evaluation Metrics

Common metrics provided by the "okahu" evaluator:

- `sentiment` - positive / negative / neutral
- `bias` - biased / unbiased
- `toxicity` - toxic / non_toxic
- `frustration` - ok / frustrated
- `hallucination` - hallucination / no_hallucination
- `contextual_relevancy` - highly_relevant / relevant / not_relevant
- `contextual_precision` - high_precision / medium_precision / low_precision
- `conversation_completeness` - complete / incomplete
- `summarization` - excellent / good / poor

Additional metrics are available. Consult the evaluator documentation for a complete list.

## 3. Running Tests

**Via VS Code Testing UI:**
1. Open Testing view (beaker icon or `Ctrl+Shift+T`)
2. VS Code auto-discovers pytest tests
3. Click ▶️ to run tests
4. Passing tests show green checkmarks

**Via Command Line:**
```bash
pytest test_evals.py              # Run all tests
pytest test_evals.py -v           # Verbose output
pytest test_evals.py::test_name   # Run specific test
pytest -k "sentiment"             # Run matching keyword
```

Passing tests display `PASSED` in green.

## 4. Understanding Failures

When an evaluation fails, the assertion error indicates the metric, expected value, and actual value:

```
AssertionError: Evaluation 'sentiment' did not match expected result. 
Expected positive. Received negative.
```

## Notes

- Always call `with_evaluation()` before `check_eval()` to configure the evaluator.
- Use `run_agent_async()` or `run_agent()` to execute your agent and generate spans before evaluation.
- Multiple evaluations can be chained: `.check_eval("sentiment", "positive").check_eval("bias", "unbiased")`
- Test files and functions must start with `test_` for pytest discovery.
- Evaluation results are sent to the Monocle UI portal when properly configured.
