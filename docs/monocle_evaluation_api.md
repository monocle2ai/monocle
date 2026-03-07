# Monocle Evaluation API Usage

This document explains how to use the Monocle evaluation quality, cost, and performance testing utilities with the `monocle_trace_asserter` fixture for validating AI agent quality, safety, and performance.

## Prerequisites

```bash
pip install monocle_test_tools
```

**Required Environment Variables:**
- `OKAHU_API_KEY` - Must be set to run evaluations

The evaluation tool will not run without this environment variable configured.

**Optional Environment Variables:**
 - `MONOCLE_EXPORTER` - set to `okahu` or `file,okahu` if you want traces to export and evaluation results to persist
    - Defaults to `file` which maintains local trace and evals only

 These are recommended, but not required environment variables

## 1. `monocle_trace_asserter` (Pytest Fixture)
A pytest fixture for running agents and asserting on evaluation metrics across entire traces or filtered spans.

**Example (Trace-level evaluation):**
```python
import pytest
from monocle_test_tools.pytest_plugin import monocle_trace_asserter

@pytest.mark.asyncio
async def test_sentiment_bias_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk",
        "Book a flight from San Jose to Seattle for 27th Nov 2025."
    )
    
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("sentiment", expected_eval="positive")\
        .check_eval("bias", expected_eval="unbiased")
```

**Example (Filtered agent evaluation):**
```python
@pytest.mark.asyncio
async def test_agent_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    monocle_trace_asserter.called_agent("flight_booking_agent")
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("conversation_completeness", expected_eval="complete")
```

**Example (Filtered tool evaluation):**
```python
@pytest.mark.asyncio
async def test_tool_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    monocle_trace_asserter.called_tool("book_flight", "flight_booking_agent")
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("toxicity", expected_eval="non_toxic")\
        .check_eval("contextual_relevancy", expected_eval="highly_relevant")
```

## 2. Available Evaluation Metrics

All metrics provided by the "okahu" evaluator:

| Metric | Possible Values |
|--------|-----------------|
| `ai_tone` | not_useful / slightly_useful / very_useful |
| `answer_relevancy` | yes / no / idk |
| `argument_correctness` | correct / incorrect / partially_correct |
| `bias` | unbiased / biased / potentially_biased |
| `contextual_precision` | high_precision / medium_precision / low_precision |
| `contextual_recall` | high_recall / medium_recall / low_recall |
| `contextual_relevancy` | highly_relevant / moderately_relevant / slightly_relevant / irrelevant |
| `conversation_completeness` | complete / mostly_complete / partially_complete / incomplete |
| `frustration` | frustrated / ok |
| `hallucination` | no_hallucination / minor_hallucination / major_hallucination |
| `knowledge_retention` | excellent_retention / good_retention / poor_retention / no_retention |
| `mcp_task_completion` | completed / partially_completed / failed / not_attempted |
| `misuse` | no_misuse / potential_misuse / clear_misuse |
| `offtopic` | on_topic / off_topic |
| `pii_leakage` | no_pii / potential_pii / pii_leakage |
| `role_adherence` | excellent_adherence / good_adherence / poor_adherence / no_adherence |
| `sentiment` | negative / positive / neutral |
| `summarization` | excellent / good / fair / poor |
| `toxicity` | non_toxic / mildly_toxic / moderately_toxic / highly_toxic |

## 3. Using Positive and Negative Expectations

The `check_eval()` method supports both **positive expectations** (values that should match) and **negative expectations** (values that should NOT match) through separate parameters:

- `expected_eval`: Values the evaluation result should match
- `unexpected_eval`: Values the evaluation result should NOT match

Both parameters are optional, but **at least one must be provided**. Each accepts either:
- **String**: For a single value
- **List of strings**: For multiple values

The method validates that there's no overlap between `expected_eval` and `unexpected_eval`.

### Syntax

**Single positive value:**
```python
# Must be "positive"
.check_eval("sentiment", expected_eval="positive")
```

**Single negative value:**
```python
# Must NOT be "negative"
.check_eval("sentiment", unexpected_eval="negative")
```

**Multiple positive values:**
```python
# Must be "positive" OR "neutral"
.check_eval("sentiment", expected_eval=["positive", "neutral"])
```

**Multiple negative values:**
```python
# Must NOT be "highly_toxic" AND NOT "moderately_toxic"
.check_eval("toxicity", unexpected_eval=["highly_toxic", "moderately_toxic"])
```

**Combined positive and negative expectations:**
```python
# Must be "positive" or "neutral" AND must NOT be "negative"
.check_eval("sentiment", expected_eval=["positive", "neutral"], unexpected_eval="negative")
```

### Examples

**Example 1: Single positive value**
```python
@pytest.mark.asyncio
async def test_positive_sentiment(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Result must be "positive"
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("sentiment", expected_eval="positive")
```

**Example 2: Single negative value**
```python
@pytest.mark.asyncio
async def test_not_negative_sentiment(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Result can be anything except "negative"
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("sentiment", unexpected_eval="negative")
```

**Example 3: Multiple negative values**
```python
@pytest.mark.asyncio
async def test_no_toxicity(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Ensure no toxicity at all
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("toxicity", unexpected_eval=["mildly_toxic", "moderately_toxic", "highly_toxic"])
```

**Example 4: Combined positive and negative expectations**
```python
@pytest.mark.asyncio
async def test_task_completion(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Task must be completed or partially_completed, AND must not have failed
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("mcp_task_completion", 
                   expected_eval=["completed", "partially_completed"], 
                   unexpected_eval="failed")
```

## 4. Running Tests

**Via VS Code Testing UI:**
1. Open Testing view (beaker icon or `Ctrl+Shift+T`)
2. VS Code auto-discovers pytest tests
3. Click ▶️ to run tests
4. Passing tests show green checkmarks while failing tests show red X's

**Via Command Line:**
```bash
pytest test_evals.py              # Run all tests
pytest test_evals.py -v           # Verbose output
pytest test_evals.py::test_name   # Run specific test
pytest -k "sentiment"             # Run matching keyword
```

Passing tests display `PASSED` in green.

## 5. Understanding Failures

When an evaluation fails, the assertion error indicates the metric, expected value, and actual value:

```
AssertionError: Evaluation 'sentiment' did not match expected result. 
Expected positive. Received negative.
```

## 6. Cost and Performance Testing

The `monocle_trace_asserter` fixture provides methods for validating performance metrics such as token consumption and execution time. These assertions help ensure your AI agents stay within acceptable cost and latency boundaries.

### Token Limit Validation

**`under_token_limit(token_limit: int)`** - Asserts that the total token usage across all inference spans is under the specified limit. This includes both prompt tokens and completion tokens.

**Example:**
```python
@pytest.mark.asyncio
async def test_token_usage(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Assert total token usage does not exceed 1500 tokens
    monocle_trace_asserter.under_token_limit(1500)
```

### Duration Limit Validation

**`under_duration(duration_limit: float)`** - Asserts that the workflow execution duration is under the specified time limit in seconds. The duration is measured from the workflow span (root span with name "workflow"). Supports decimal values for precise timing requirements.

**Note:** Duration assertions must be called on the full trace and cannot be used after filtering operations like `called_tool()` or `called_agent()`, as these filters exclude the spans needed for duration measurement.

**Examples:**
```python
@pytest.mark.asyncio
async def test_execution_duration(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Assert workflow completes within 10 seconds
    monocle_trace_asserter.under_duration(10)
    
    # Decimal values supported for precise timing
    monocle_trace_asserter.under_duration(12.5)
```

**Chaining Performance Assertions:**
```python
@pytest.mark.asyncio
async def test_cost_and_performance(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Chain token and duration assertions (both work on full trace)
    monocle_trace_asserter\
        .under_token_limit(1500)\
        .under_duration(10)
```

**Note:** Duration assertions cannot be chained after `called_tool()` or `called_agent()` filters, but can be combined with token limit assertions.

**Understanding Performance Failures:**

When performance assertions fail, you'll see clear error messages:

```
AssertionError: Token limit exceeded: 1623 > 1500
```

```
AssertionError: Workflow duration 13.45s exceeds limit 12.5s.
```

## Notes

- Call `with_evaluation("okahu")` once per test before calling `check_eval()` to configure the evaluator.
 - You don't have to declare evaluator each time
- Use `run_agent_async()` or `run_agent()` to execute your agent and generate spans before evaluation.
- The `check_eval()` method requires at least one of `expected_eval` or `unexpected_eval` parameters:
  - `expected_eval`: Accepts a **string** or **list of strings** for values that should match
  - `unexpected_eval`: Accepts a **string** or **list of strings** for values that should NOT match
  - Both parameters can be used together for comprehensive validation
  - The method validates that there's no overlap between the two parameters
- Multiple evaluations can be chained: `.check_eval("sentiment", expected_eval="positive").check_eval("bias", expected_eval="unbiased")`
- Test files and functions must start with `test_` for pytest discovery.
- Evaluation results are sent to the Okahu UI portal when `MONOCLE_EXPORTER` is properly configured.
