# Monocle Performance, Quality, and Cost Testing API

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

### Basic Syntax

The `check_eval()` method supports two syntax styles:

**Simple syntax** (defaults to "traces" fact):
```python
.check_eval("metric_name", "expected_value")
.check_eval("metric_name", expected="value")
.check_eval("metric_name", not_expected="value")
```

**Explicit syntax** (specify fact name):
```python
.check_eval(fact_name="fact_type", eval_name="metric_name", expected="value")
.check_eval(fact_name="fact_type", eval_name="metric_name", not_expected="value")
```

### Examples

**Example 1: Basic trace-level evaluation (simple syntax)**
```python
import pytest
from monocle_test_tools.pytest_plugin import monocle_trace_asserter

@pytest.mark.asyncio
async def test_sentiment_bias_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk",
        "Book a flight from San Jose to Seattle for 27th Nov 2025."
    )
    
    # Simple syntax - evaluates on "traces" fact by default
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("sentiment", "positive")\
        .check_eval("bias", "unbiased")
```

**Example 2: Multiple evaluators in same test**
```python
@pytest.mark.asyncio
async def test_multiple_evaluators(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Use okahu evaluator
    monocle_trace_asserter.with_evaluation("okahu").check_eval("frustration", "ok")
    
    # Switch to bert_score evaluator
    monocle_trace_asserter.with_evaluation("bert_score", {"model_type": "bert-base-uncased"})
    
    # Switch back to okahu evaluator
    monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")
```

**Example 3: Custom failure message**
```python
@pytest.mark.asyncio
async def test_with_message(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("summarization", "excellent", 
                   message="Summarization should capture all key details accurately.")
```

## 2. Understanding Fact Names

**Fact names** represent the scope or type of spans that evaluations are performed on. When you call `check_eval()`, you can optionally specify which fact type to evaluate:

### Available Fact Names

- **`traces`** (default) - Evaluates the entire trace from start to finish
- **`agent_sessions`** - Evaluates individual agent session spans
- **`conversations`** - Evaluates conversation-level interactions
- **`inferences`** - Evaluates individual LLM inference calls
- **`generic`** - Generic span evaluation (supports sentiment, offtopic)
- **`assessment`** - Assessment-level evaluation (supports offtopic)

### When to Specify Fact Names

**Omit `fact_name` (defaults to "traces"):**
```python
# Simple syntax - evaluates on entire trace
.check_eval("sentiment", "positive")
.check_eval("bias", "unbiased")
```

**Specify `fact_name` explicitly:**
```python
# Evaluate sentiment on each inference span
.check_eval(fact_name="inferences", eval_name="sentiment", not_expected="negative")

# Evaluate hallucination on agent sessions
.check_eval(fact_name="agent_sessions", eval_name="hallucination", expected="no_hallucination")

# Evaluate frustration on conversations
.check_eval(fact_name="conversations", eval_name="frustration", expected="ok")
```

**Important:** Each metric supports specific fact names. Using an unsupported combination will raise an error. Refer to the "Supported Fact Names" column in the table below to see which fact names each metric supports.

## 3. Available Evaluation Metrics

All metrics provided by the "okahu" evaluator:

| Metric | Possible Values | Supported Fact Names |
|--------|-----------------|----------------------|
| `ai_tone` | not_useful / slightly_useful / very_useful | traces |
| `answer_relevancy` | yes / no / idk | traces |
| `argument_correctness` | correct / incorrect / partially_correct | traces |
| `bias` | unbiased / biased / potentially_biased | traces |
| `contextual_precision` | high_precision / medium_precision / low_precision | traces |
| `contextual_recall` | high_recall / medium_recall / low_recall | traces |
| `contextual_relevancy` | highly_relevant / moderately_relevant / slightly_relevant / irrelevant | traces, agent_sessions |
| `conversation_completeness` | complete / mostly_complete / partially_complete / incomplete | traces |
| `frustration` | frustrated / ok | traces, conversations |
| `hallucination` | no_hallucination / minor_hallucination / major_hallucination | traces, agent_sessions |
| `knowledge_retention` | excellent_retention / good_retention / poor_retention / no_retention | traces |
| `mcp_task_completion` | completed / partially_completed / failed / not_attempted | traces, agent_sessions |
| `misuse` | no_misuse / potential_misuse / clear_misuse | traces, agent_sessions |
| `offtopic` | on_topic / off_topic | conversations, assessment, generic |
| `pii_leakage` | no_pii / potential_pii / pii_leakage | traces, agent_sessions |
| `role_adherence` | excellent_adherence / good_adherence / poor_adherence / no_adherence | traces, agent_sessions |
| `sentiment` | negative / positive / neutral | traces, conversations, agent_sessions, inferences, generic |
| `summarization` | excellent / good / fair / poor | traces |
| `toxicity` | non_toxic / mildly_toxic / moderately_toxic / highly_toxic | traces, agent_sessions |

## 4. Using Positive and Negative Expectations

The `check_eval()` method supports both **positive expectations** (values that should match) and **negative expectations** (values that should NOT match) through separate parameters:

- `expected`: **(Optional)** Values the evaluation result should match
- `not_expected`: **(Optional)** Values the evaluation result should NOT match

Both parameters are optional, but **at least one must be provided** — omitting both will raise a `ValueError`.

Each parameter accepts either:
- **String**: For a single value
- **List of strings**: For multiple values

**Important:** The method validates that there's no overlap between `expected` and `not_expected`. If the same value appears in both parameters, a `ValueError` will be raised. This ensures the expectations are mutually exclusive and logically consistent.

**Invalid example (will raise ValueError):**
```python
# ERROR: "positive" appears in both expected and not_expected
.check_eval("sentiment", 
           expected=["positive", "neutral"], 
           not_expected="positive")  # ValueError!
```

### Syntax Options

| Syntax Type | Example | Description |
|-------------|---------|-------------|
| **Simple - Single value** | `.check_eval("sentiment", "positive")` | Positional syntax, defaults to traces fact |
| **Simple - Expected** | `.check_eval("sentiment", expected="positive")` | Single expected value, defaults to traces fact |
| **Simple - Not expected** | `.check_eval("sentiment", not_expected="negative")` | Single not_expected value, defaults to traces fact |
| **Simple - Multiple expected** | `.check_eval("sentiment", expected=["positive", "neutral"])` | Multiple expected values (OR logic), defaults to traces fact |
| **Simple - Multiple not expected** | `.check_eval("toxicity", not_expected=["highly_toxic", "moderately_toxic"])` | Multiple not_expected values (AND logic), defaults to traces fact |
| **Simple - Both parameters** | `.check_eval("sentiment", expected=["positive", "neutral"], not_expected="negative")` | Combine expected and not_expected, defaults to traces fact |
| **Explicit - With fact_name** | `.check_eval(fact_name="inferences", eval_name="sentiment", not_expected="negative")` | Specify fact type explicitly |
| **Explicit - Multiple expected** | `.check_eval(fact_name="agent_sessions", eval_name="sentiment", expected=["positive", "neutral"])` | Multiple expected values on specific fact |
| **Explicit - Multiple not expected** | `.check_eval(fact_name="agent_sessions", eval_name="toxicity", not_expected=["highly_toxic", "moderately_toxic"])` | Multiple not_expected values on specific fact |
| **Explicit - Both parameters** | `.check_eval(fact_name="conversations", eval_name="sentiment", expected=["positive", "neutral"], not_expected="negative")` | Combine expected and not_expected on specific fact |

### Comprehensive Examples

**Example 1: Simple syntax - Trace-level evaluation (defaults to traces)**
```python
@pytest.mark.asyncio
async def test_trace_sentiment_bias(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Simple syntax - evaluates on traces fact by default
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval("sentiment", "positive")\
        .check_eval("bias", "unbiased")
```

**Example 2: Inference spans with not_expected**
```python
@pytest.mark.asyncio
async def test_inferences_sentiment(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Evaluate sentiment on each inference span - must NOT be negative
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval(fact_name="inferences", eval_name="sentiment", not_expected="negative")
```

**Example 3: Conversations with multiple evaluations**
```python
@pytest.mark.asyncio
async def test_conversations_evaluation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Evaluate conversation-level metrics
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval(fact_name="conversations", eval_name="frustration", not_expected="frustrated")
    monocle_trace_asserter.check_eval(fact_name="conversations", eval_name="offtopic", expected="on_topic")
```

**Example 4: Agent sessions with combined expected and not_expected**
```python
@pytest.mark.asyncio
async def test_agent_sessions_quality(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Role adherence must be excellent or good, NOT poor or none
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval(fact_name="agent_sessions", eval_name="role_adherence",
                   expected=["excellent_adherence", "good_adherence"],
                   not_expected=["poor_adherence", "no_adherence"])
    
    # Task must be completed or partially completed, NOT failed
    monocle_trace_asserter.check_eval(fact_name="agent_sessions", eval_name="mcp_task_completion", 
                   expected=["completed", "partially_completed"], not_expected="failed")
```

**Example 5: Multiple not_expected values**
```python
@pytest.mark.asyncio
async def test_toxicity_check(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Agent sessions must NOT be toxic at any level
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval(fact_name="agent_sessions", eval_name="toxicity", 
                   not_expected=["highly_toxic", "moderately_toxic", "mildly_toxic"])
```

**Example 6: Generic and assessment fact types**
```python
@pytest.mark.asyncio
async def test_generic_and_assessment(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Evaluate with generic fact type
    monocle_trace_asserter.with_evaluation("okahu")\
        .check_eval(fact_name="generic", eval_name="sentiment", 
                   expected=["positive", "neutral"], not_expected="negative")
    
    # Evaluate with assessment fact type
    monocle_trace_asserter.check_eval(fact_name="assessment", eval_name="offtopic", expected="on_topic")
```

## 5. Running Tests

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

## 6. Understanding Failures

When an evaluation fails, the assertion error indicates the metric, expected value, and actual value:

```
AssertionError: Evaluation 'sentiment' did not match expected result. 
Expected positive. Received negative.
```

## 7. Cost and Performance Testing

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

**`under_duration(duration_limit: float, units: str = "seconds", span_type: str = "workflow")`** - Asserts that span durations are under the specified time limit. The method measures duration for different span types based on the `span_type` parameter.

**Parameters:**
- `duration_limit`: Maximum duration allowed (float, supports decimal values)
- `units`: Time unit for the limit - `"seconds"` (default), `"ms"`, or `"minutes"`
- `span_type`: Type of spans to measure - `"workflow"` (default), `"agent_invocation"`, `"tool_invocation"`, `"agent_turn"`, or `"inference"`

**Supported Span Types:**
- `"workflow"` - Root workflow spans (overall execution time)
- `"agent_invocation"` - Agent invocation spans (individual agent executions)
- `"tool_invocation"` - Tool invocation spans (individual tool executions)
- `"agent_turn"` - Agent turn spans (turn-level interactions)
- `"inference"` - Inference spans (LLM API calls)

**Examples:**

**Basic workflow duration:**
```python
@pytest.mark.asyncio
async def test_execution_duration(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Assert workflow completes within 10 seconds
    monocle_trace_asserter.under_duration(10, units="seconds", span_type="workflow")
    
    # Decimal values supported for precise timing
    monocle_trace_asserter.under_duration(12.5, units="seconds", span_type="workflow")
```

**Multiple fact types with different units:**
```python
@pytest.mark.asyncio
async def test_multi_level_duration(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Workflow under 10 seconds
    monocle_trace_asserter.under_duration(10, units="seconds", span_type="workflow")
    # Each agent invocation under 0.2 minutes
    monocle_trace_asserter.under_duration(0.2, units="minutes", span_type="agent_invocation")
    # Each inference under 5000 milliseconds
    monocle_trace_asserter.under_duration(5000, units="ms", span_type="inference")
```

**Filtered span duration with called_agent() and called_tool():**
```python
@pytest.mark.asyncio
async def test_filtered_duration(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Filter to specific agent and check its invocation duration
    monocle_trace_asserter.called_agent("flight_booking_agent")\
        .under_duration(0.2, units="minutes", span_type="agent_invocation")
    
    # Filter to specific tool and check its invocation duration
    monocle_trace_asserter.called_tool("book_flight")\
        .under_duration(5000, units="ms", span_type="tool_invocation")
```

**Chaining Performance Assertions:**
```python
@pytest.mark.asyncio
async def test_cost_and_performance(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Chain token and duration assertions
    monocle_trace_asserter\
        .under_token_limit(1500)\
        .under_duration(10, units="seconds", span_type="workflow")\
        .under_duration(0.2, units="minutes", span_type="agent_turn")\
        .under_duration(4000, units="ms", span_type="inference")
```

**Multiple filtered assertions:**
```python
@pytest.mark.asyncio
async def test_multiple_filtered_limits(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", "query")
    
    # Check flight booking agent duration
    monocle_trace_asserter.called_agent("flight_booking_agent")\
        .under_duration(0.15, units="minutes", span_type="agent_invocation")
    
    # Check hotel booking agent duration and tokens
    monocle_trace_asserter.called_agent("hotel_booking_agent")\
        .under_duration(0.18, units="minutes", span_type="agent_invocation")\
        .under_token_limit(900)
    
    # Check tool invocation duration and tokens
    monocle_trace_asserter.called_tool("book_flight")\
        .under_duration(5000, units="ms", span_type="tool_invocation")\
        .under_token_limit(80)
```

**Understanding Performance Failures:**

When performance assertions fail, you'll see clear error messages with entity-specific information:

```
AssertionError: Token limit exceeded: 1623 > 1500
```

```
AssertionError: Duration limit exceeded: workflow took 13.45 seconds (limit: 12.5 seconds)
```

```
AssertionError: Duration limit exceeded for agent 'flight_booking_agent': agent_invocation took 0.23 minutes (limit: 0.2 minutes)
```

```
AssertionError: Duration limit exceeded for tool 'book_flight': tool_invocation took 5.5 ms (limit: 5 ms)
```

## Notes

### Evaluator Configuration
- Call `with_evaluation("okahu")` to configure an evaluator before calling `check_eval()`
- You don't have to declare the evaluator each time — it persists for subsequent `check_eval()` calls
- You can switch evaluators within the same test by calling `with_evaluation()` with a different evaluator name
- Example: `.with_evaluation("bert_score", {"model_type": "bert-base-uncased"})`

### Agent Execution
- Use `run_agent_async()` or `run_agent()` to execute your agent and generate spans before evaluation

### check_eval() Method Signatures

**Simple syntax (defaults to "traces" fact):**
```python
check_eval(eval_name, expected=None, not_expected=None, message=None)
```

**Explicit syntax (with fact_name):**
```python
check_eval(fact_name, eval_name, expected=None, not_expected=None, message=None)
```

**Parameters:**
- `fact_name`: **(Optional)** The fact type to evaluate (traces, agent_sessions, conversations, inferences, generic, assessment). Defaults to "traces" if omitted.
- `eval_name`: **(Required)** The metric/evaluation name (e.g., "sentiment", "bias", "hallucination")
- `expected`: **(Optional)** Accepts a **string** or **list of strings** for values that should match
- `not_expected`: **(Optional)** Accepts a **string** or **list of strings** for values that should NOT match
- `message`: **(Optional)** Custom message to display on evaluation failure

**Important:**
- At least one of `expected` or `not_expected` must be provided — omitting both will raise a `ValueError`
- Both parameters can be used together for comprehensive validation
- The method validates that there's no overlap between `expected` and `not_expected`
- Each metric supports specific fact names — using an unsupported combination will raise an error (see "Supported Fact Names" column in the metrics table)

### Chaining and Pytest
- Multiple evaluations can be chained: `.check_eval("sentiment", "positive").check_eval("bias", "unbiased")`
- Test files and functions must start with `test_` for pytest discovery
- Evaluation results are sent to the Okahu UI portal when `MONOCLE_EXPORTER` is properly configured
