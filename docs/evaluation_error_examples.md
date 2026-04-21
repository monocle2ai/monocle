# Evaluation Error Examples

This document captures example evaluation failures for different fact types and evaluation templates. These examples help understand evaluation behavior and expected results.

---

## Traces (fact_name: traces)

### sentiment Evaluation

**Test Case**: `test_trace_level_sentiment_bias_evaluation`

**Test Code**:
```python
monocle_trace_asserter.with_evaluation("okahu").check_eval('sentiment', 'positive')
```

**Failure Details**:
- **Expected**: `positive`
- **Received**: `neutral`
- **Location**: `test_tools\tests\integration\test_evals.py#11`

**Explanation from Evaluator**:
> The conversation revolves around booking flights and hotels without expressing any emotional content. The user repeatedly asks for flight bookings, and the agent responds in a neutral, factual manner. Therefore, the sentiment is classified as neutral.

**Key Insight**: 
Trace-level sentiment evaluation requires explicit emotional content. Business-oriented conversations about bookings are typically classified as neutral even if the service is being provided successfully. Positive sentiment requires more than just successful interaction.

### toxicity Evaluation - Service Error

**Test Case**: `test_filtered_agent_tool_evaluation`

**Test Code**:
```python
monocle_trace_asserter.check_eval('toxicity', 'non_toxic')\
    .check_eval('contextual_relevancy', 'highly_relevant')
```

**Failure Details**:
- **Error Type**: HTTP 500 - Service Error
- **Location**: `test_tools\tests\integration\test_evals.py#34`

**Error Message**:
> Evaluation service returned HTTP 500: <empty body>

**Key Insight**: 
Internal server errors indicate backend processing issues in the evaluation service. This can occur when the service is not running properly, traces aren't fully ingested, or specific evaluation templates have processing errors.

---

## Inferences (fact_name: inferences)

### sentiment Evaluation - Empty Result

**Test Case**: `test_v1_inferences_sentiment_evaluation`

**Test Code**:
```python
monocle_trace_asserter.with_evaluation("okahu").check_eval(
    fact_name='inferences', 
    eval_name='sentiment', 
    not_expected='negative'
)
```

**Failure Details**:
- **Error Type**: Empty result array
- **Location**: `test_tools\tests\integration\test_evals.py#50`

**Error Message**:
> Unexpected response format from evaluation service. Expected 'result' key in response. Received: {'message': 'Job submitted', 'job_id': 'interactive_598bcbb8e34a4a3db48ee7f2d7cbcbef_dev_1774549147', 'result': []}

**Key Insight**: 
When the evaluation service returns an empty `result` array, it means no matching facts were found for evaluation. This typically happens when:
- No inference spans exist in the trace
- The fact_name doesn't match any spans in the data
- Timing issues where evaluation runs before data is fully indexed

### frustration Evaluation - Template Not Found

**Test Case**: `test_v1_invalid_template_wrong_fact_name_frustration`

**Test Code**:
```python
monocle_trace_asserter.with_evaluation("okahu").check_eval(
    fact_name='inferences', 
    eval_name='frustration', 
    expected='ok'
)
```

**Failure Details**:
- **Error Type**: Template not found for fact type
- **Location**: `test_tools\tests\integration\test_evals.py#108`

**Error Message**:
> Evaluation template with name 'frustration' and group_by 'inferences' not found in Okahu. Available templates: [{'name': 'sentiment', 'group_by': 'inferences'}]

**Key Insight**: 
Only specific evaluation templates are available for each fact type. For `inferences`, only `sentiment` is available. The `frustration` template is available for other fact types but not for inferences.

---

## Agent Requests (fact_name: agent_requests)

### offtopic Evaluation

**Test Case**: `test_v1_agent_requests_evaluation`

**Test Code**:
```python
monocle_trace_asserter.with_evaluation("okahu").check_eval(
    fact_name='agent_requests', 
    eval_name='offtopic', 
    expected='on_topic'
)
```

**Failure Details**:
- **Expected**: `on_topic`
- **Received**: `off_topic`
- **Location**: `test_tools\tests\integration\test_evals.py#59`

**Explanation from Evaluator**:
> The response in turn 2 does not accurately address the user's original request in turn 1. The user specifically requested to book a flight, but the assistant expands the response to include a hotel booking as well. While booking a flight is within the scope of typical travel assistance, discussing additional options or services not explicitly requested can lead to ambiguity. This response could lack clarity on whether the user intended to include a hotel booking, thus deviating from the primary task of booking a flight. There is no clear indication that the assistant confirmed this additional request with the user before acting, which could imply overstepping its designated purpose. Therefore, while related to travel assistance, this expansion into additional services deviates from the exact nature of the user's request, making it off-topic.

**Key Insight**: 
The `offtopic` evaluation on `agent_requests` is strict about scope adherence. Even though hotel booking is related to travel, suggesting services beyond what the user explicitly requested (flight booking only) is considered off-topic. The evaluator expects agents to stick closely to the user's stated intent without expanding scope unless confirmed by the user.

---

## Conversations (fact_name: conversations)

### hallucination Evaluation - Template Not Found

**Test Case**: `test_v1_invalid_template_wrong_fact_name`

**Test Code**:
```python
monocle_trace_asserter.with_evaluation("okahu").check_eval(
    fact_name='conversations', 
    eval_name='hallucination', 
    expected='no_hallucination'
)
```

**Failure Details**:
- **Error Type**: Template not found for fact type
- **Location**: `test_tools\tests\integration\test_evals.py#99`

**Error Message**:
> Evaluation template with name 'hallucination' and group_by 'conversations' not found in Okahu. Available templates: [{'name': 'frustration', 'group_by': 'conversations'}, {'name': 'offtopic', 'group_by': 'conversations'}, {'name': 'sentiment', 'group_by': 'conversations'}]

**Key Insight**: 
For `conversations` fact type, only `frustration`, `offtopic`, and `sentiment` evaluation templates are available. The `hallucination` template exists for other fact types but not for conversations.

---

## Template Validation Errors

### Nonexistent Template

**Test Case**: `test_v1_invalid_template_nonexistent`

**Test Code**:
```python
monocle_trace_asserter.with_evaluation("okahu").check_eval(
    fact_name='traces', 
    eval_name='code_quality', 
    expected='excellent'
)
```

**Failure Details**:
- **Error Type**: Template does not exist
- **Location**: `test_tools\tests\integration\test_evals.py#88`

**Error Message**:
> Evaluation template with name 'code_quality' and group_by 'traces' not found in Okahu. Available templates: [{'name': 'argument_correctness', 'group_by': 'traces'}, {'name': 'contextual_precision', 'group_by': 'traces'}, {'name': 'answer_relevancy', 'group_by': 'traces'}, {'name': 'summarization', 'group_by': 'traces'}, {'name': 'pii_leakage', 'group_by': 'traces'}, {'name': 'role_adherence', 'group_by': 'traces'}, {'name': 'contextual_recall', 'group_by': 'traces'}, {'name': 'sentiment', 'group_by': 'traces'}, {'name': 'toxicity', 'group_by': 'traces'}, {'name': 'knowledge_retention', 'group_by': 'traces'}, {'name': 'contextual_relevancy', 'group_by': 'traces'}, {'name': 'frustration', 'group_by': 'traces'}, {'name': 'conversation_completeness', 'group_by': 'traces'}, {'name': 'mcp_task_completion', 'group_by': 'traces'}, {'name': 'bias', 'group_by': 'traces'}, {'name': 'hallucination', 'group_by': 'traces'}, {'name': 'misuse', 'group_by': 'traces'}]

**Key Insight**: 
The evaluation service validates that the template exists before submitting jobs. `code_quality` is not a valid evaluation template. Check available templates for each fact type before creating test assertions.

**Available Templates for 'traces' fact:**
- argument_correctness
- contextual_precision
- answer_relevancy
- summarization
- pii_leakage
- role_adherence
- contextual_recall
- sentiment
- toxicity
- knowledge_retention
- contextual_relevancy
- frustration
- conversation_completeness
- mcp_task_completion
- bias
- hallucination
- misuse

---

## Common Error Patterns

### 1. Empty Result Arrays
When `result: []` is returned, it means no facts matching the criteria were found. Check:
- Correct fact_name for the span types in your trace
- Spans have the required attributes (e.g., `scope.agentic.turn` for agent_requests)
- Data has been ingested before evaluation runs

### 2. Template Not Found for Fact Type
Templates are fact-specific. A template available for one fact type may not be available for another. Always verify template availability for your specific fact_name.

### 3. HTTP 500 Errors
Backend service errors with empty bodies indicate:
- Service not running or misconfigured
- Internal processing errors in the evaluation service
- Race conditions with data ingestion

### 4. Sentiment Classification
Sentiment evaluations distinguish between:
- **Positive**: Explicit positive emotional content
- **Neutral**: Factual, business-oriented interactions without emotional tone
- **Negative**: Frustration, complaints, or negative emotional content

---

## Notes

- Evaluation results can be sensitive to exact phrasing and scope boundaries
- Agent responses should confirm with users before expanding beyond the original request scope
- Different fact types support different evaluation templates
- Always validate template availability before creating assertions
- Empty result arrays indicate no matching facts were found for evaluation
