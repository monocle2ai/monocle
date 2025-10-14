# Enhanced Monocle Agent Testing Framework

A comprehensive pytest-style assertion framework for testing AI agents based on traces, with advanced plugin system and LLM-powered analysis capabilities.

## Key Features

🎯 **Intuitive pytest-style API** - Familiar pytest fixtures and base classes  
🔍 **Fluent trace assertions** - Chain assertions for readable test code  
🧩 **Plugin Architecture** - Extensible assertion system with HTTP, Agent, Core, and LLM plugins  
🤖 **LLM-Powered Analysis** - Ask natural language questions about trace data with structured JSON responses  
📊 **Agent Flow Validation** - Test complex multi-agent workflows and interactions  
🌐 **HTTP Testing** - Complete REST API testing suite with semantic similarity validation  
🚀 **Simple setup** - Minimal boilerplate, maximum functionality  
📝 **Rich error messages** - Clear feedback when tests fail



## Quick Start

```python
import pytest
from monocle_tfwk import BaseAgentTest

class TestMyAgent(BaseAgentTest):
    @pytest.fixture
    def agent(self):
        return MyAgent()
    
    def test_basic_functionality(self, agent):
        result = agent.run("Book flight to Mumbai")
        
        # Chain assertions for readable test code
        (self.assert_traces()
         .has_agent("travel_agent")
         .called_tool("book_flight")
         .contains_input("Mumbai")
         .with_output_containing("booked"))
```
        assert "expected output" in result
```

### Fluent Assertion API

The `TraceAssertions` class provides a fluent API for validating agent traces:

```python
# Chain assertions for readable test code
(self.assert_trace()
 .has_agent("travel_agent")
 .called_tool("book_flight")
 .with_input_containing("Mumbai")
 .with_output_containing("booked"))
```

## LLM-Powered Trace Analysis

The framework now includes powerful LLM analysis capabilities that let you ask natural language questions about your traces and get structured JSON responses.

### Basic LLM Analysis

```python
@pytest.mark.asyncio
async def test_booking_confirmation_with_llm(self, travel_agent):
    """Test booking confirmation using LLM analysis."""
    result = await travel_agent.process_travel_request("Book a hotel in Mumbai")
    
    traces = self.assert_traces()
    
    # Ask LLM to analyze the traces and return structured JSON
    confirmation = await traces.ask_llm_about_traces(
        "Is the hotel booking confirmed? "
        "Return your answer in JSON format with this exact structure: "
        '{"confirmed": true/false, "reason": "brief explanation"}'
    )
    
    # Parse and validate the structured response
    import json
    confirmation_data = json.loads(confirmation)
    assert confirmation_data["confirmed"], f"Hotel booking should be confirmed: {confirmation}"
```

### Advanced LLM Queries

```python
# Cost analysis
cost_analysis = await traces.ask_llm_about_traces(
    "What is the total cost mentioned in the traces? "
    "Return JSON: {'total_cost': number, 'currency': 'string', 'breakdown': ['item1', 'item2']}"
)

# Error detection
error_analysis = await traces.ask_llm_about_traces(
    "Were there any errors or failures? "
    "Return JSON: {'has_errors': true/false, 'error_details': ['list of errors'], 'severity': 'low|medium|high'}"
)

# Performance analysis
perf_analysis = await traces.ask_llm_about_traces(
    "Analyze the performance metrics. "
    "Return JSON: {'total_duration_ms': number, 'slow_operations': ['list'], 'recommendations': ['list']}"
)
```

## Available Assertions

### Agent Flow Assertions

```python
# Define participants for multi-agent workflows  
participants = [
    ("U", "User", "actor", "Human user initiating requests"),
    ("TC", "Travel_Coordinator", "agent", "Main orchestration agent"),
    ("FA", "Flight_Assistant", "agent", "Flight booking specialist"),
    ("BFT", "book_flight_tool", "tool", "Flight booking tool")
]

# Comprehensive agentic flow validation
whole_flow = {
    "participants": participants,
    "required": ["TC"],  # Required participants
    "interactions": [
        ("U", "TC", "request"),      # User requests Travel Coordinator  
        ("TC", "FA", "delegation"),  # TC delegates to Flight Assistant
        ("FA", "BFT", "invocation")  # Flight Assistant calls booking tool
    ]
}

# Validate complete agentic workflow
self.assert_traces().assert_agent_flow(whole_flow)

# Individual agent assertions
self.assert_trace().has_agent("agent_name")
self.assert_traces().assert_agent_called("travel_coordinator")
self.assert_traces().assert_agent_type("agent.openai_agents")
self.assert_traces().assert_workflow_complete()

# Flow pattern assertions
self.assert_traces().assert_flow("agent.reasoning -> tool_use -> response")
self.assert_traces().assert_conditional_flow("coordinator", "needs_booking", 
                                           ["booking_agent"], ["info_agent"])

# HTTP testing
self.assert_traces().assert_get_requests(min_count=2)
self.assert_traces().assert_http_status_code(200)

# Tool assertions
self.assert_trace().called_tool("tool_name")

# Input/output assertions
self.assert_trace().with_input_containing("search text")
self.assert_traces().semantically_contains_output("booking confirmed", threshold=0.8)

# Span and attribute assertions 
self.assert_trace().has_spans(min_count=1)
self.assert_trace().has_attribute("model", "gpt-4")

# Filtering
self.assert_trace().filter_by_name("llm.call")
```

## Plugin Architecture

The framework uses a plugin system with Core, HTTP, Agent, LLM, and Semantic plugins for comprehensive testing.

## Advanced Usage Examples

```python
# Parametrized testing for multiple scenarios
@pytest.mark.parametrize("destination", ["Mumbai", "Delhi", "Goa"])
def test_booking_destinations(self, agent, destination):
    result = agent.run(f"Book flight to {destination}")
    self.assert_traces().contains_input(destination)

# Performance testing with timing constraints
def test_agent_performance(self, agent):
    agent.run("Complex booking request")
    self.assert_traces().assert_total_duration_under(5.0)

# Custom validation helpers
def assert_booking_workflow_complete(self, traces):
    """Custom helper for booking validation."""
    (traces.has_spans(min_count=3)
     .called_tool("book_flight")
     .semantically_contains_output("confirmed", threshold=0.8))
```

## Key Features

✅ **Intuitive** - Uses familiar pytest patterns  
✅ **Less boilerplate** - No complex test case definitions  
✅ **Better debugging** - Clear failure descriptions with `debug_spans()`  
✅ **Flexible** - Easy to combine with custom assertions

## Troubleshooting

### Plugin Loading Errors
Ensure plugins are imported in `plugins/__init__.py`

### LLM JSON Issues  
Framework uses OpenAI structured JSON mode for clean parsing

Use `debug_spans()` to inspect actual vs expected patterns

## Best Practices

1. **Use descriptive test names** and chain assertions logically  
2. **Leverage LLM analysis** for complex validation with `ask_llm_about_traces()`
3. **Test multi-agent flows** with `assert_agent_flow()` for workflow validation
4. **Use semantic similarity** to test LLM output meaning, not exact strings
5. **Test performance** with timing assertions like `assert_api_call_timing()`

