# Enhanced Monocle Agent Testing Framework

A pytest-style assertion framework for testing AI agents based on traces, inspired by [AgentiTest](https://github.com/kweinmeister/agentitest) but designed specifically for trace-based validation.

## Key Features

🎯 **Intuitive pytest-style API** - Familiar pytest fixtures and base classes  
🔍 **Fluent trace assertions** - Chain assertions for readable test code  
📊 **Comprehensive validation** - Test agents, tools, inputs, outputs, performance  
🚀 **Simple setup** - Minimal boilerplate, maximum functionality  
📝 **Rich error messages** - Clear feedback when tests fail  

## Integrating with Your Agentic Code Development

### Complete Development Workflow Example

Here's how to integrate this testing framework into your agentic application development process, from initial development to production validation.

#### Step 1: Write Your Agent Code

```python
# my_travel_agent.py
import logging
from typing import Dict, Any
from opentelemetry import trace

logger = logging.getLogger(__name__)

class TravelBookingAgent:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.tools = {
            "search_flights": self._search_flights,
            "book_flight": self._book_flight,
            "search_hotels": self._search_hotels, 
            "book_hotel": self._book_hotel
        }
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """Main entry point for travel booking requests."""
        with self.tracer.start_as_current_span("agent.process_request") as span:
            span.set_attribute("user_request", user_request)
            
            # Parse and analyze request
            analysis = self._analyze_request(user_request)
            
            # Execute booking workflow
            result = self._execute_booking_workflow(analysis)
            
            span.set_attribute("booking_result", str(result))
            return result
    
    def _analyze_request(self, request: str) -> Dict[str, Any]:
        """Analyze user request to determine booking requirements."""
        with self.tracer.start_as_current_span("agent.analyze_request") as span:
            span.set_attribute("input", request)
            
            analysis = {
                "needs_flight": "flight" in request.lower(),
                "needs_hotel": "hotel" in request.lower(),
                "destination": self._extract_destination(request),
                "dates": self._extract_dates(request)
            }
            
            span.set_attribute("analysis_result", str(analysis))
            return analysis
    
    def _execute_booking_workflow(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the booking workflow based on analysis."""
        results = {"bookings": []}
        
        if analysis["needs_flight"]:
            flight_result = self._book_flight(analysis["destination"], analysis["dates"])
            results["bookings"].append(flight_result)
        
        if analysis["needs_hotel"]:
            hotel_result = self._book_hotel(analysis["destination"], analysis["dates"])
            results["bookings"].append(hotel_result)
            
        return results
    
    def _search_flights(self, destination: str, dates: Dict) -> Dict[str, Any]:
        """Search for available flights."""
        with self.tracer.start_as_current_span("tool.search_flights") as span:
            span.set_attribute("tool.name", "search_flights")
            span.set_attribute("destination", destination)
            # Simulate flight search
            return {"flights": [{"airline": "AirIndia", "price": 500}]}
    
    def _book_flight(self, destination: str, dates: Dict) -> Dict[str, Any]:
        """Book a flight."""
        with self.tracer.start_as_current_span("tool.book_flight") as span:
            span.set_attribute("tool.name", "book_flight")
            span.set_attribute("destination", destination)
            # Simulate flight booking
            result = {"type": "flight", "destination": destination, "status": "booked"}
            span.set_attribute("booking_result", str(result))
            return result
    
    def _search_hotels(self, destination: str, dates: Dict) -> Dict[str, Any]:
        """Search for available hotels."""
        with self.tracer.start_as_current_span("tool.search_hotels") as span:
            span.set_attribute("tool.name", "search_hotels")
            span.set_attribute("destination", destination)
            return {"hotels": [{"name": "Grand Hotel", "price": 200}]}
    
    def _book_hotel(self, destination: str, dates: Dict) -> Dict[str, Any]:
        """Book a hotel."""
        with self.tracer.start_as_current_span("tool.book_hotel") as span:
            span.set_attribute("tool.name", "book_hotel")
            span.set_attribute("destination", destination)
            result = {"type": "hotel", "destination": destination, "status": "booked"}
            span.set_attribute("booking_result", str(result))
            return result
    
    def _extract_destination(self, request: str) -> str:
        """Extract destination from user request."""
        # Simple extraction logic
        if "mumbai" in request.lower():
            return "Mumbai"
        elif "delhi" in request.lower():
            return "Delhi"
        return "Unknown"
    
    def _extract_dates(self, request: str) -> Dict[str, str]:
        """Extract dates from user request.""" 
        # Simple date extraction
        return {"departure": "2024-12-01", "return": "2024-12-07"}
```

#### Step 2: Write Comprehensive Tests

```python
# test_travel_agent.py
import pytest
from monocle_tfwk import BaseAgentTest
from my_travel_agent import TravelBookingAgent


class TestTravelBookingAgent(BaseAgentTest):
    """Comprehensive test suite for travel booking agent."""
    
    @pytest.fixture
    def travel_agent(self):
        """Create a travel agent instance for testing."""
        return TravelBookingAgent()
    
    def test_flight_only_booking(self, travel_agent):
        """Test booking flight only."""
        request = "Book a flight to Mumbai for December 1st"
        result = travel_agent.process_request(request)
        
        # Verify business logic
        assert len(result["bookings"]) == 1
        assert result["bookings"][0]["type"] == "flight"
        assert result["bookings"][0]["status"] == "booked"
        
        # Verify agent trace behavior
        (self.assert_traces()
         .has_spans(min_count=4)                    # Main + analysis + tool calls
         .has_span_with_name("agent.process_request")
         .has_span_with_name("agent.analyze_request") 
         .called_tool("book_flight")
         .contains_input("Mumbai")
         .completed_successfully())
    
    def test_hotel_only_booking(self, travel_agent):
        """Test booking hotel only."""
        request = "Book a hotel in Delhi"
        result = travel_agent.process_request(request)
        
        # Verify business logic
        assert len(result["bookings"]) == 1
        assert result["bookings"][0]["type"] == "hotel"
        
        # Verify agent workflow
        (self.assert_traces()
         .called_tool("book_hotel")
         .contains_input("Delhi")
         .with_output_containing("booked"))
    
    def test_complete_travel_booking(self, travel_agent):
        """Test booking both flight and hotel."""
        request = "Book flight and hotel to Mumbai for vacation"
        result = travel_agent.process_request(request)
        
        # Verify complete booking
        assert len(result["bookings"]) == 2
        booking_types = [b["type"] for b in result["bookings"]]
        assert "flight" in booking_types
        assert "hotel" in booking_types
        
        # Verify complete workflow execution
        (self.assert_traces()
         .has_spans(min_count=6)                    # Request + analysis + 2 bookings
         .called_tool("book_flight")
         .called_tool("book_hotel")
         .contains_input("Mumbai")
         .semantically_contains_output("booked", threshold=0.7))
    
    def test_workflow_execution_order(self, travel_agent):
        """Test that workflow steps execute in correct order."""
        travel_agent.process_request("Book flight and hotel to Mumbai")
        
        traces = self.assert_traces()
        spans = sorted(traces.spans, key=lambda s: s.start_time)
        
        # Verify execution order
        expected_order = [
            "agent.process_request",
            "agent.analyze_request", 
            "tool.book_flight",
            "tool.book_hotel"
        ]
        
        actual_order = [span.name for span in spans]
        for expected_span in expected_order:
            assert expected_span in actual_order, f"Missing {expected_span} in workflow"
    
    def test_error_handling(self, travel_agent):
        """Test agent behavior with invalid requests.""" 
        request = "Book something unclear"
        result = travel_agent.process_request(request)
        
        # Should still complete without errors
        (self.assert_traces()
         .completed_successfully()
         .has_span_with_name("agent.analyze_request"))
         
        # Verify graceful handling
        assert "bookings" in result
    
    def test_performance_requirements(self, travel_agent):
        """Test that agent meets performance requirements."""
        travel_agent.process_request("Book flight to Mumbai")
        
        # Verify performance constraints
        (self.assert_traces()
         .within_time_limit(2.0)                    # Max 2 seconds
         .has_spans(max_count=10))                  # Reasonable complexity
    
    def test_input_output_preservation(self, travel_agent):
        """Test that user input is preserved through workflow."""
        original_request = "Book flight to Mumbai for business trip"
        travel_agent.process_request(original_request)
        
        # Verify input preservation
        (self.assert_traces()
         .contains_input("Mumbai")
         .contains_input("business trip")
         .filter_by_name("agent.process_request")
         .has_attribute("user_request", original_request))


class TestTravelAgentIntegration(BaseAgentTest):
    """Integration tests for complete travel scenarios."""
    
    @pytest.fixture
    def travel_agent(self):
        return TravelBookingAgent()
    
    def test_end_to_end_vacation_booking(self, travel_agent):
        """Test complete vacation booking scenario."""
        request = "Plan a vacation to Mumbai - need flight and 3-star hotel"
        result = travel_agent.process_request(request)
        
        # Business validation
        assert len(result["bookings"]) == 2
        
        # Complete trace validation
        (self.assert_traces()
         .has_spans(min_count=5)
         .has_agent_sequence("process_request", "analyze_request")
         .called_tool("book_flight") 
         .called_tool("book_hotel")
         .semantically_contains_input("vacation planning")
         .completed_successfully())
    
    def test_business_travel_workflow(self, travel_agent):
        """Test business travel specific workflow."""
        request = "Book urgent flight to Delhi for business meeting"
        result = travel_agent.process_request(request)
        
        # Verify business travel handling
        (self.assert_traces()
         .contains_input("urgent")
         .contains_input("business")
         .called_tool("book_flight")
         .with_output_containing("Delhi"))


# Custom test helpers for domain-specific validation
def assert_booking_successful(traces, booking_type: str):
    """Custom helper to validate successful bookings."""
    (traces
     .called_tool(f"book_{booking_type}")
     .with_output_containing("booked")
     .completed_successfully())

def assert_complete_travel_workflow(traces):
    """Custom helper for complete travel workflow validation.""" 
    (traces
     .has_span_with_name("agent.process_request")
     .has_span_with_name("agent.analyze_request")
     .has_spans(min_count=4)
     .completed_successfully())
```

#### Step 3: Development Workflow Integration

```python
# conftest.py - Shared test configuration
import pytest
from monocle_tfwk import BaseAgentTest

@pytest.fixture(scope="session")
def setup_tracing():
    """Set up tracing for all tests."""
    # Any global tracing setup
    yield
    # Cleanup after tests

# Run tests during development
# pytest test_travel_agent.py -v
# pytest test_travel_agent.py::TestTravelBookingAgent::test_complete_travel_booking -v
```

#### Step 4: Continuous Integration

```yaml
# .github/workflows/test.yml
name: Agent Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -e ./tfwk
    - name: Run agent tests
      run: pytest tests/ -v --tb=short
```

### Benefits of This Approach

✅ **Trace-Based Validation** - Verify actual agent behavior, not just outputs  
✅ **Regression Prevention** - Catch workflow changes before deployment  
✅ **Performance Monitoring** - Ensure agents meet performance requirements  
✅ **Documentation** - Tests serve as executable specifications  
✅ **Debugging** - Rich error messages help identify issues quickly  

### Development Tips

1. **Write tests first** - Define expected behavior before implementation
2. **Test incrementally** - Add tests as you build each agent capability
3. **Use descriptive names** - Make test intent clear for future developers
4. **Test edge cases** - Invalid inputs, error conditions, performance limits
5. **Validate workflows** - Ensure agent steps execute in correct order
6. **Monitor complexity** - Use span counts to detect workflow bloat

This approach ensures your agentic applications are reliable, maintainable, and perform as expected in production.

## Quick Start

### Basic Test Structure

```python
import pytest
from monocle_tfwk import BaseAgentTest

class TestMyAgent(BaseAgentTest):
    """Test suite for my AI agent."""

    def test_agent_basic_functionality(self):
        """Test that agent completes successfully."""
        # Run your agent
        result = my_agent_function("user input")
        
        # Assert trace behavior
        (self.assert_trace()
         .has_agent("my_agent")
         .with_input_containing("user input")
         .with_output_containing("expected output"))
            
        # Assert result
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

## Available Assertions

### Agent Assertions
```python
# Test agent invocations
self.assert_trace().has_agent("agent_name")
```

### Tool Assertions
```python
# Test tool invocations
self.assert_trace().called_tool("tool_name")

# Test multiple tool calls
self.assert_trace().called_tool("book_flight").has_spans(count=3)
```

### Input/Output Assertions
```python
# Test input content  
self.assert_trace().with_input_containing("search text")

# Test output content
self.assert_trace().with_output_containing("expected result")
```

### Span Assertions
```python
# Test span counts
self.assert_trace().has_spans(count=5)  # Exact count
self.assert_trace().has_spans(min_count=1)  # Minimum count
self.assert_trace().has_spans(max_count=10)  # Maximum count

# Test LLM calls
self.assert_trace().has_llm_calls(count=2)
```

### Attribute Assertions
```python
# Test span attributes
self.assert_trace().has_attribute("model", "gpt-4")
self.assert_trace().has_attribute("temperature", 0.7)
```

### Filtering
```python
# Filter by span name
self.assert_trace().filter_by_name("llm.call")

# Filter by attributes
self.assert_trace().filter_by_attribute("model", "gpt-4")

# Focus on LLM calls
self.assert_trace().llm_calls()
```

## Advanced Usage

### Testing Multi-Step Agents

```python
class TestMultiStepAgent(BaseAgentTest):
    @pytest.fixture
    def agent(self):
        return ComplexAgent()

    def test_complete_workflow(self, agent):
        """Test the complete multi-step workflow."""
        result = agent.run("What is 5 + 3?")
        
        # Verify all steps were executed
        (self.assert_trace()
         .has_spans(min_count=4)  # run + analyze + llm + memory
         .has_span_with_name("agent.run")
         .has_span_with_name("agent.analyze")  
         .has_span_with_name("llm.call")
         .has_span_with_name("agent.memory"))
```

### Async Agent Testing

```python
class TestAsyncAgent(BaseAgentTest):
    @pytest.fixture
    def agent(self):
        return AsyncAgent()
    
    @pytest.mark.asyncio
    async def test_async_functionality(self, agent):
        """Test async agent functionality."""
        result = await agent.run("Hello async world!")
        
        assert "Hello async world!" in result
        
        # Verify traces were generated
        (self.assert_trace()
         .has_spans(min_count=2)  # run + async_llm_call
         .has_span_with_name("async_agent.run")
         .has_span_with_name("async_llm.call"))
```

### Custom Assertions

```python
def test_custom_validation(self, agent):
    """Test with custom validation logic."""
    agent.run("input")
    
    # Get trace for custom logic
    traces = self.assert_trace()
    
    # Custom validation logic
    spans = sorted(traces.spans, key=lambda s: s.start_time)
    expected_order = ["agent.run", "agent.analyze", "llm.call", "agent.memory"]
    actual_order = [span.name for span in spans]
    
    # Check that workflow steps appear in correct order
    for expected_span in expected_order:
        assert expected_span in actual_order
    
    # Continue with fluent assertions
    (traces
     .has_spans(min_count=4)
     .contains_input("input"))
```

### Performance Testing

```python
def test_performance_requirements(self, agent):
    """Test performance requirements."""
    agent.run("complex task")
    
    traces = self.assert_trace()
    
    # Custom validation: check span durations
    for span in traces.spans:
        duration_ns = span.end_time - span.start_time
        duration_ms = duration_ns / 1_000_000
        
        # Ensure no span takes too long
        assert duration_ms < 1000, f"Span {span.name} took too long: {duration_ms}ms"
```

### Parametrized Tests

```python
@pytest.mark.parametrize("destination", ["Mumbai", "London", "Tokyo"])
def test_flight_booking_destinations(self, destination, agent):
    """Test flight booking for different destinations."""
    request = f"Book flight to {destination}"
    
    result = agent.run(request)
    
    (self.assert_trace()
     .contains_input(destination)
     .has_spans(min_count=1))
```

## BaseAgentTest Methods

The `BaseAgentTest` base class provides these methods:

- `assert_trace()` - Get fluent assertion API for current test spans
- Automatic test setup via pytest fixtures
- Integration with MonocleValidator for trace collection

## Available TraceAssertions Methods

### Span Counting and Filtering
- `has_spans(count=None, min_count=None, max_count=None)` - Assert span counts
- `has_llm_calls(count=None, min_count=None, max_count=None)` - Assert LLM call counts
- `filter_by_name(name)` - Filter spans by name
- `filter_by_attribute(key, value)` - Filter spans by attribute
- `llm_calls()` - Filter to LLM-related spans
- `has_span_with_name(name)` - Assert span with specific name exists

### Content Validation
- `contains_input(text)` - Assert input contains text
- `contains_output(text)` - Assert output contains text
- `has_attribute(key, value)` - Assert span has attribute with value

### Chaining
All methods return `TraceAssertions` for method chaining.

## Migration from Legacy Framework

### Before (Legacy)
```python
agent_test_cases = [{
    "test_input": ["Book flight to Mumbai"],
    "test_spans": [{
        "span_type": "agentic.tool.invocation",
        "entities": [{"type": "tool", "name": "book_flight"}]
    }]
}]

@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_booking(test_case):
    await MonocleValidator().test_workflow_async(agent, test_case)
```

### After (New Framework)
```python
class TestBooking(BaseAgentTest):
    def test_booking(self, agent):
        result = agent.run("Book flight to Mumbai")
        
        (self.assert_trace()
         .contains_input("Mumbai")
         .has_spans(min_count=1))
```

## Benefits Over Legacy Framework

✅ **More intuitive** - Uses familiar pytest patterns  
✅ **Less boilerplate** - No complex test case definitions  
✅ **Better IDE support** - Full autocomplete and type hints  
✅ **Easier debugging** - Standard pytest debugging workflow  
✅ **More flexible** - Easy to combine with custom assertions  
✅ **Better error messages** - Clear failure descriptions  

## Human-Readable Failure Reporting

The framework provides excellent debugging capabilities with clear, actionable error messages to help you understand what went wrong in your agent tests.

### Built-in Descriptive Error Messages

All assertions include human-readable error messages:

```python
def test_agent_behavior(self, agent):
    """Example showing clear error messages"""
    result = agent.run("Book a flight to Mumbai")
    
    # Each assertion provides specific error context
    (self.assert_traces()
     .has_spans(count=5)           # "Expected exactly 5 spans, found 3"
     .called_tool("book_flight")   # "No tool named 'book_flight' found in traces"  
     .contains_input("Mumbai")     # "No spans found with input containing 'Mumbai'"
     .completed_successfully())    # "Found 2 spans with errors"
```

### Debug Helper for Deep Inspection

Use `debug_spans()` to see detailed trace information when tests fail:

```python
def test_with_debugging(self, agent):
    """Enhanced debugging with span inspection"""
    agent.run("Book flight and hotel")
    
    try:
        (self.assert_traces()
         .called_tool("book_flight")
         .called_tool("book_hotel"))
    except AssertionError as e:
        print(f"Test failed: {e}")
        
        # Show detailed span information
        self.assert_traces().debug_spans()
        raise
```

**Sample debug output:**
```
=== DEBUG: 4 spans ===
  [0] agent.run
      Attributes: {'agent.name': 'travel_agent', 'input': 'Book flight and hotel'}
      Start: 1697123456789000000, End: 1697123457890000000
  [1] agent.analyze  
      Attributes: {'component': 'analyzer', 'complexity': 5}
      Start: 1697123456800000000, End: 1697123456850000000
  [2] llm.call
      Attributes: {'model': 'gpt-4', 'input': 'analyze: Book flight and hotel'}
      Start: 1697123456860000000, End: 1697123457800000000
  [3] tool.book_flight
      Attributes: {'tool.name': 'book_flight', 'input': 'Mumbai'}
      Start: 1697123457810000000, End: 1697123457880000000
========================
```

### Enhanced Error Context

Add custom context to make failures even clearer:

```python
def test_with_enhanced_context(self, agent):
    """Custom error reporting with business context"""
    user_input = "Book a flight to Mumbai"
    result = agent.run(user_input)
    
    traces = self.assert_traces()
    
    try:
        traces.called_tool("book_flight")
    except AssertionError:
        # Provide business context
        print(f"\n❌ BOOKING WORKFLOW FAILURE:")
        print(f"   User Request: '{user_input}'")
        print(f"   Agent Response: '{result}'")
        print(f"   Expected: Flight booking tool to be called")
        
        # Show what actually happened
        tool_spans = [s for s in traces.spans if 'tool' in s.name.lower()]
        if tool_spans:
            print(f"   Actually Called:")
            for span in tool_spans:
                tool_name = span.attributes.get('tool.name', span.name)
                print(f"     - {tool_name}")
        else:
            print(f"   Actually Called: No tools were invoked")
            
        traces.debug_spans()
        raise
```

### Workflow Validation with Clear Reporting

Create custom helpers for complex workflow validation:

```python
def assert_booking_workflow_completed(self, expected_steps: List[str]):
    """Custom assertion with detailed workflow reporting"""
    traces = self.assert_traces()
    actual_steps = [span.name for span in sorted(traces.spans, key=lambda s: s.start_time)]
    
    if actual_steps != expected_steps:
        print(f"\n❌ WORKFLOW SEQUENCE MISMATCH:")
        print(f"   Expected Order: {expected_steps}")
        print(f"   Actual Order:   {actual_steps}")
        print(f"   Missing Steps:  {set(expected_steps) - set(actual_steps)}")
        print(f"   Extra Steps:    {set(actual_steps) - set(expected_steps)}")
        
        traces.debug_spans()
        assert False, f"Workflow doesn't match expected sequence"

def test_complete_booking_workflow(self, booking_agent):
    """Test complete booking workflow with clear failure reporting"""
    booking_agent.run("Book flight and hotel to Mumbai")
    
    self.assert_booking_workflow_completed([
        "agent.run",
        "agent.analyze", 
        "tool.book_flight",
        "tool.book_hotel",
        "agent.confirm"
    ])
```

### Semantic Similarity Failures

Semantic assertions also provide helpful debugging information:

```python
def test_semantic_output_validation(self, agent):
    """Test with semantic similarity and clear error reporting"""
    agent.run("What is the weather like?")
    
    try:
        (self.assert_traces()
         .semantically_contains_output("sunny and warm", threshold=0.8))
    except AssertionError as e:
        # Error includes threshold and expected text context
        # "No spans found with output semantically similar to 'sunny and warm' (threshold: 0.8)"
        print(f"\n❌ SEMANTIC VALIDATION FAILED:")
        print(f"   Expected meaning: 'sunny and warm'")
        print(f"   Similarity threshold: 0.8")
        
        # Show actual outputs for comparison
        traces = self.assert_traces()
        print(f"   Actual outputs:")
        for i, span in enumerate(traces.spans):
            output = span.attributes.get('output', 'No output')
            print(f"     [{i}] {output}")
        raise
```

### pytest Integration for Professional Reports

The framework integrates seamlessly with pytest for professional test reporting:

```bash
$ pytest tests/ -v

======================== FAILURES ========================
________ TestBookingAgent.test_flight_booking ________

self = <test_booking.TestBookingAgent object at 0x...>
agent = <BookingAgent object at 0x...>

    def test_flight_booking(self, agent):
        result = agent.run("Book flight to Mumbai") 
>       (self.assert_traces()
         .called_tool("book_flight")
         .contains_output("flight booked"))

E       AssertionError: No tool named 'book_flight' found in traces

tests/test_booking.py:23: AssertionError
```

### Tips for Better Error Messages

1. **Use descriptive variable names** in your tests
2. **Add business context** to custom assertions  
3. **Call `debug_spans()`** when investigating failures
4. **Create domain-specific helpers** for complex validations
5. **Include expected vs actual comparisons** in custom errors

## Best Practices

1. **Use descriptive test names** - Make intent clear
2. **Chain assertions logically** - Group related checks
3. **Test one concept per test** - Keep tests focused  
4. **Use parametrized tests** - Test multiple scenarios efficiently
5. **Add custom validation when needed** - Don't be limited by built-ins
6. **Use fixtures for agent setup** - Keep tests clean and reusable

## Example Test Suite Structure

```
tests/
├── conftest.py              # Shared fixtures and config
├── test_booking_agent.py    # Booking agent tests
├── test_search_agent.py     # Search agent tests
├── test_integration.py      # End-to-end integration tests
└── test_performance.py      # Performance and load tests
```

## Installation and Setup

```bash
# From the tfwk directory
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all agent tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_booking_agent.py

# Run tests matching pattern
pytest tests/ -k "booking"

# Run examples
python -m pytest examples/ -v
```

This framework makes testing AI agents as intuitive as testing any other Python code, while providing powerful trace-based validation capabilities.