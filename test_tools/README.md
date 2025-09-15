# Monocle Test Tools

A comprehensive testing and validation framework for monocle AI agent tracing. This package provides tools for validating agent behavior, tool invocations, inference responses, and overall AI system performance.

## Features

- **Agent Validation**: Verify that specific agents are invoked and delegate tasks correctly
- **Tool Validation**: Ensure tools are called with expected inputs and produce expected outputs  
- **Response Evaluation**: Validate agent responses using configurable evaluation metrics (BERT Score, custom evaluators)
- **Inference Testing**: Test model inference responses against expected schemas or content
- **Performance Monitoring**: Check token limits, error states, and warnings
- **Telemetry Integration**: Built on OpenTelemetry for comprehensive observability

## Installation

```bash
pip install monocle_test_tools
```

For development:
```bash
pip install monocle_test_tools[dev]
```

## Quick Start

```python
import pytest
from monocle_test_tools import MonocleValidator, TestCase, AgentToVerify

# Initialize the validator
validator = MonocleValidator()

# Define test cases
test_cases = [
    TestCase(
        test_case_name="test_agent_response",
        response_to_evaluate="The weather is sunny today",
        agents=[
            AgentToVerify(
                name="weather_agent",
                response_to_evaluate="sunny weather information"
            )
        ]
    )
]

# Use as pytest decorator
@validator.monocle_testcase(test_cases)
def test_my_agent_system(test_case):
    # Your agent system code here
    # The validator will automatically capture spans and validate
    agent_response = call_your_agent_system(test_case.input)
    return agent_response
```

## Core Components

### TestCase
Defines what to validate in your test:

```python
TestCase(
    test_case_name="comprehensive_test",
    test_description="Test agent delegation and tool usage",
    input="What's the weather like?",
    response_to_evaluate="Expected response content",
    agents=[AgentToVerify(name="weather_agent")],
    tools=[ToolToVerify(name="weather_api", expected_input="location")],
    inferences=[VerifyInference(expected_schema=response_schema)],
    check_errors=True,
    check_warnings=True
)
```

### AgentToVerify
Validate specific agent behavior:

```python
AgentToVerify(
    name="assistant_agent",
    response_to_evaluate="Expected response pattern",
    from_agent="coordinator_agent"  # Check delegation
)
```

### ToolToVerify  
Validate tool invocations:

```python
ToolToVerify(
    name="database_query",
    from_agent="data_agent",
    expected_input="SELECT * FROM users",
    expected_output="Query results"
)
```

### VerifyInference
Validate model inference responses:

```python
# Schema validation
VerifyInference(
    expected_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"}
        }
    },
    max_output_tokens=150
)

# Pydantic model validation
from pydantic import BaseModel

class ResponseModel(BaseModel):
    answer: str
    confidence: float

VerifyInference(
    expected_schema=ResponseModel,
    response_to_evaluate="Expected content"
)
```

## Custom Evaluators

Create custom evaluation logic by extending `BaseEval`:

```python
from monocle_test_tools.evals import BaseEval

class CustomEval(BaseEval):
    def get_eval(self, eval_name: str, eval_args: dict) -> dict:
        if eval_name == "custom_metric":
            # Your custom evaluation logic
            return {"valid_response": True, "score": 0.95}
        return super().get_eval(eval_name, eval_args)

# Use with validator
validator = MonocleValidator()
test_cases = [TestCase(...)]

@validator.monocle_testcase(test_cases, eval=CustomEval())
def test_with_custom_eval(test_case):
    # Your test code
    pass
```

## Async Support

The framework supports both sync and async test functions:

```python
@validator.monocle_testcase(test_cases)
async def test_async_agent(test_case):
    response = await async_agent_call(test_case.input)
    return response
```

## Integration with pytest

The framework integrates seamlessly with pytest:

```bash
# Run all tests
pytest

# Run with specific markers
pytest -m "agent and not slow"

# Run with coverage
pytest --cov=test_tools --cov-report=html
```

## Configuration

Configure evaluation thresholds and behavior:

```python
from monocle_test_tools.evals import DefaultEval

custom_eval = DefaultEval(
    bert_scorer_acceptible_f1_threshold=0.8,
    bert_scorer_precision_threshold=0.75,
    bert_scorer_recall_threshold=0.7,
    model_type='bert-large-uncased'
)
```

## Dependencies

- **pytest**: Test framework integration
- **OpenTelemetry**: Telemetry and span collection  
- **monocle_apptrace**: Core monocle tracing functionality
- **bert-score**: Default response evaluation
- **transformers**: NLP model support
- **pydantic**: Schema validation
- **jsonschema**: JSON schema validation

## License

Apache License 2.0 - see LICENSE file for details.

## Contributing

Please refer to the main monocle repository for contribution guidelines.
