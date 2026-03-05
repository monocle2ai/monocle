# Monocle Test Assertions

This document provides a comprehensive reference for all assertions supported by the Monocle fluent API.

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
| `under_duration()` | `duration_limit` | **Asserts:** The workflow execution time is under the specified limit (in seconds). Fails if duration exceeds limit. |
| `check_eval()` | `eval_name`, `expected_eval`, `fact_name="traces"` | **Asserts:** The evaluation result matches the expected value. Fails if evaluation result differs from expected. |

## Configuration Methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| `with_evaluation()` | `eval`, `eval_options=None` | Configures the evaluation method for input/output comparisons (e.g., LLM-based) |
| `with_comparer()` | `comparer` | Sets the comparison strategy for input/output matching (e.g., exact, token-based) |

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
