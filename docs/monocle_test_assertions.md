# Monocle Test Assertions

This document provides a comprehensive reference for all assertions supported by the Monocle fluent API.

> **Note:** All assertion methods support an optional custom error message parameter. You can provide your own error message to make test failures more descriptive and easier to debug.

## Agent & Tool Calls

| Method | Parameters | Description |
|--------|-----------|-------------|
| `called_tool()` | `tool_name`, `agent_name=None` | **Asserts:** The specified tool was called (optionally by a specific agent). Fails if the tool was not invoked. |
| `does_not_call_tool()` | `tool_names`, `agent_name=None` | **Asserts:** The specified tool was NOT called (optionally by a specific agent). Fails if the tool was invoked. |
| `called_agent()` | `agent_name` | **Asserts:** The specified agent was called during the workflow. Fails if the agent was not invoked. |
| `does_not_call_agent()` | `agent_name` | **Asserts:** The specified agent was NOT called during the workflow. Fails if the agent was invoked. |

## Performance & Evaluation

| Method | Parameters | Description |
|--------|-----------|-------------|
| `under_token_limit()` | `token_limit` | **Asserts:** The total token count across all spans is below the specified limit. Fails if limit is exceeded. |
| `under_duration()` | `duration_limit`, `units="seconds"`, `span_type="workflow"` | **Asserts:** Span durations are under the specified limit. `units`: `seconds` (default), `ms`, `minutes`. Fails if duration exceeds limit. |
| `check_eval()` | `eval_name=None`, `expected=None`, `not_expected=None`, `fact_name="traces"`, `template_path=None`, `template=None`, `min_facts=1`, `fail_threshold=0`, `max_facts=None` | **Asserts:** The evaluation result matches (`expected`) / avoids (`not_expected`) the given label(s). Provide **exactly one** of `eval_name` (Okahu template), `template_path` (custom-template JSON file), or `template` (inline dict). `min_facts`/`fail_threshold`/`max_facts` apply only in time-window (filtered) mode. See the [evaluation API guide](monocle_evaluation_api.md) for filtered evaluation and result-report accessors. |

## Configuration Methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| `with_evaluation()` | `eval`, `eval_options=None` | Configures the evaluation method for input/output comparisons (e.g., LLM-based) |
| `with_comparer()` | `comparer` | Sets the comparison strategy for input/output matching (e.g., exact, token-based) |

## Attribute, Event, and Generic Assertions

| Method | Parameters | Description |
|--------|-----------|-------------|
| `has_attribute()` | `attribute_name`, `expected=None` | **Asserts:** A span carries the given attribute (and, when `expected` is provided, that it equals `expected`). Narrows the filtered spans to the matches for further chaining. Fails if no span has the attribute/value. |
| `does_not_have_attribute()` | `attribute_name`, `expected=None` | **Asserts:** No span carries the given attribute (or the given attribute/value pair when `expected` is provided). Fails if a matching span is found. |
| `has_event()` | `event_name`, `attribute_name=None`, `expected=None` | **Asserts:** A span has an event named `event_name` (and, when `attribute_name`/`expected` are provided, that the event carries a matching attribute). Narrows the filtered spans to the matches for further chaining. Fails if no span has the event/attribute. |
| `where()` | `attribute=None`, `event=None`, `predicate=None` | **Generic selector.** Narrows the filtered spans to those matching **all** given criteria (AND-ed): `attribute` is a `{name: expected}` mapping (`expected=None` checks presence), `event` is `{"name": <event_name>, "attributes": {<attr>: <expected>}}` (the `"attributes"` key is optional), and `predicate` is a `Callable[[Span], bool]`. Fails if no span matches. `has_attribute`/`has_event` are thin wrappers over this. |
| `does_not_match()` | `attribute=None`, `event=None`, `predicate=None` | **Asserts:** No span matches all the given criteria (same arguments as `where`). Fails if a matching span is found. |

### Generic `where()` example

```python
# Assert some span is an agentic turn that also emitted 1000 total tokens
monocle_trace_asserter.where(
    attribute={"span.type": "agentic.turn"},
    event={"name": "metadata", "attributes": {"total_tokens": 1000}},
)

# Arbitrary matching with a predicate
monocle_trace_asserter.where(predicate=lambda span: span.attributes.get("entity.count", 0) >= 2)
```

## Input Assertions

| Method | Parameters | Description |
|--------|-----------|-------------|
| `has_input()` | `expected_input` | **Asserts:** The input exactly matches the expected value. Fails if input does not match. |
| `has_any_input()` | `*expected_inputs` | **Asserts:** The input matches at least one of the provided expected values. Fails if input matches none. |
| `does_not_have_input()` | `unexpected_input` | **Asserts:** The input does NOT match the specified value. Fails if input matches. |
| `does_not_have_any_input()` | `*unexpected_inputs` | **Asserts:** The input matches none of the specified values. Fails if input matches any. |
| `contains_input()` | `expected_input_substring` | **Asserts:** The input contains the specified substring. Fails if substring is not found. |
| `contains_any_input()` | `*expected_input_substrings` | **Asserts:** The input contains at least one of the specified substrings. Fails if none are found. |
| `does_not_contain_input()` | `unexpected_input_substring` | **Asserts:** The input does NOT contain the specified substring. Fails if substring is found. |
| `does_not_contain_any_input()` | `*unexpected_input_substrings` | **Asserts:** The input contains none of the specified substrings. Fails if any are found. |

## Output Assertions

| Method | Parameters | Description |
|--------|-----------|-------------|
| `has_output()` | `expected_output` | **Asserts:** The output exactly matches the expected value. Fails if output does not match. |
| `has_any_output()` | `*expected_outputs` | **Asserts:** The output matches at least one of the provided expected values. Fails if output matches none. |
| `does_not_have_output()` | `unexpected_output` | **Asserts:** The output does NOT match the specified value. Fails if output matches. |
| `does_not_have_any_output()` | `*unexpected_outputs` | **Asserts:** The output matches none of the specified values. Fails if output matches any. |
| `contains_output()` | `expected_output_substring` | **Asserts:** The output contains the specified substring. Fails if substring is not found. |
| `contains_any_output()` | `*expected_output_substrings` | **Asserts:** The output contains at least one of the specified substrings. Fails if none are found. |
| `does_not_contain_output()` | `unexpected_output_substring` | **Asserts:** The output does NOT contain the specified substring. Fails if substring is found. |
| `does_not_contain_any_output()` | `*unexpected_output_substrings` | **Asserts:** The output contains none of the specified substrings. Fails if any are found. |

## Scope Assertions

Scopes are contextual tags attached to spans (stored as the `scope.<name>` span attribute). These assertions check the value of a named scope across the currently filtered spans. A positive assertion passes when **at least one** filtered span satisfies it.

| Method | Parameters | Description |
|--------|-----------|-------------|
| `has_scope()` | `scope_name`, `expected_value` (optional) | **Asserts:** A span has the scope with the expected value. If `expected_value` is omitted, only the scope's presence is checked. Fails if no span matches. |
| `has_any_scope()` | `scope_name`, `*expected_values` | **Asserts:** The scope value matches at least one of the provided values. Fails if it matches none. |
| `does_not_have_scope()` | `scope_name`, `unexpected_value` (optional) | **Asserts:** No span has the scope with the specified value. If `unexpected_value` is omitted, the scope must be entirely absent. Fails if any span matches. |
| `does_not_have_any_scope()` | `scope_name`, `*unexpected_values` | **Asserts:** No span has the scope with any of the specified values. Fails if any span matches. |
| `contains_scope()` | `scope_name`, `expected_substring` | **Asserts:** The scope value contains the substring (case-insensitive). Fails if not found. |
| `contains_any_scope()` | `scope_name`, `*expected_substrings` | **Asserts:** The scope value contains at least one of the substrings (case-insensitive). Fails if none are found. |
| `does_not_contain_scope()` | `scope_name`, `unexpected_substring` | **Asserts:** The scope value does NOT contain the substring. Fails if found. |
| `does_not_contain_any_scope()` | `scope_name`, `*unexpected_substrings` | **Asserts:** The scope value contains none of the substrings. Fails if any are found. |
