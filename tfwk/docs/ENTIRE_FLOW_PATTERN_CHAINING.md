# Entire Flow Pattern Chaining Guide

## Overview

The Monocle Testing Framework provides comprehensive support for validating entire trace flows using pattern-based assertions. This guide demonstrates how to chain multiple patterns together to validate complex, end-to-end workflows with comprehensive trace analysis.

## Table of Contents

- [Basic Flow Chaining](#basic-flow-chaining)
- [Pattern Types](#pattern-types)
- [Advanced Pattern Combinations](#advanced-pattern-combinations)
- [Business Logic Patterns](#business-logic-patterns)
- [Performance Patterns](#performance-patterns)
- [Complete Flow Validation](#complete-flow-validation)
- [Best Practices](#best-practices)

## Basic Flow Chaining

### Simple Pattern Chain

```python
from monocle_tfwk import TraceAssertions

# Chain basic flow validation patterns
traces = TraceAssertions(spans)
(traces
    .assert_agent_sequence(["supervisor", "flight_agent", "hotel_agent"])
    .assert_workflow_pattern("fan-out", ["supervisor", "worker1", "worker2"])
    .assert_agent_called_before("flight_agent", "hotel_agent")
    .completed_successfully())
```

### Pattern Reset and Continuation

```python
# Chain with reset points for complex validations
(traces
    .filter_by_name("agentic.request")
    .assert_spans(min_count=1)
    .reset_to_all_spans()  # Reset to work with all spans again
    .assert_agent_sequence(["analyzer", "executor", "validator"])
    .completed_successfully())
```

## Pattern Types

### 1. Sequential Patterns

Validate that agents execute in a specific order:

```python
def test_sequential_workflow(self, agent):
    """Test agents execute in correct sequence."""
    agent.process_request("Plan travel itinerary")
    
    (self.assert_traces()
        .assert_agent_sequence([
            "request_processor",
            "travel_analyzer", 
            "itinerary_builder",
            "confirmation_sender"
        ])
        .completed_successfully())
```

### 2. Parallel Patterns

Validate that certain agents run concurrently:

```python
def test_parallel_execution(self, agent):
    """Test parallel agent execution."""
    agent.process_request("Book flight and hotel simultaneously")
    
    (self.assert_traces()
        .assert_agents_called_in_parallel([
            "flight_booker", 
            "hotel_booker"
        ], tolerance_ms=1000)
        .completed_successfully())
```

### 3. Fan-out Patterns

Validate supervisor dispatching to multiple workers:

```python
def test_fanout_workflow(self, agent):
    """Test fan-out coordination pattern."""
    agent.process_request("Coordinate multiple bookings")
    
    (self.assert_traces()
        .assert_workflow_pattern("fan-out", [
            "coordinator",      # Supervisor
            "flight_agent",     # Worker 1
            "hotel_agent",      # Worker 2
            "car_rental_agent"  # Worker 3
        ])
        .completed_successfully())
```

### 4. Fan-in Patterns

Validate multiple workers feeding into aggregator:

```python
def test_fanin_workflow(self, agent):
    """Test fan-in aggregation pattern."""
    agent.process_request("Collect and summarize all bookings")
    
    (self.assert_traces()
        .assert_workflow_pattern("fan-in", [
            "flight_collector",    # Worker 1
            "hotel_collector",     # Worker 2
            "summary_generator"    # Aggregator
        ])
        .completed_successfully())
```

### 5. Conditional Flow Patterns

Validate branching logic based on conditions:

```python
def test_conditional_flows(self, agent):
    """Test conditional workflow branching."""
    agent.process_request("Plan business trip with expense tracking")
    
    (self.assert_traces()
        .assert_conditional_flow(
            condition_agent="trip_analyzer",
            condition_output_contains="business",
            then_agents=["expense_tracker", "receipt_manager"],
            else_agents=["vacation_planner"]
        )
        .completed_successfully())
```

## Advanced Pattern Combinations

### Multi-Pattern Workflow Validation

```python
def test_complex_travel_workflow(self, travel_agent):
    """Test complete travel workflow with multiple patterns."""
    request = "Plan complete business trip to Mumbai - flight, hotel, recommendations"
    result = travel_agent.process_request(request)
    
    traces = self.assert_traces()
    (traces
        # 1. Initiation Pattern
        .filter_by_name("agentic.request")
        .assert_spans(min_count=1)
        .contains_input("Mumbai")
        .contains_input("business trip")
        
        # 2. Sequential Analysis Pattern  
        .reset_to_all_spans()
        .assert_agent_sequence([
            "travel_supervisor", 
            "request_analyzer",
            "booking_coordinator"
        ])
        
        # 3. Parallel Booking Pattern
        .assert_agents_called_in_parallel([
            "flight_assistant", 
            "hotel_assistant"
        ], tolerance_ms=500)
        
        # 4. Tool Usage Pattern
        .assert_agent_used_tool("flight_assistant", "book_flight")
        .assert_agent_used_tool("hotel_assistant", "book_hotel") 
        
        # 5. Fan-in Completion Pattern
        .assert_workflow_pattern("fan-in", [
            "flight_assistant",
            "hotel_assistant", 
            "trip_summarizer"
        ])
        
        # 6. Output Validation Pattern
        .semantically_contains_output("flight booked", threshold=0.7)
        .semantically_contains_output("hotel reserved", threshold=0.7)
        
        # 7. Performance Pattern
        .within_time_limit(5.0)
        .completed_successfully())
```

### JMESPath Query Pattern Chains

```python
def test_jmespath_pattern_validation(self, agent):
    """Test complex patterns using JMESPath queries."""
    agent.process_complex_workflow()
    
    traces = self.assert_traces()
    
    # Chain JMESPath queries for advanced pattern validation
    (traces
        # Agent execution pattern
        .query("[?attributes.\"agent.name\"]")
        .debug_execution_flow()
        
        # Tool usage mapping pattern
        .query("""
            [?contains(name, 'tool_invocation')].{
                agent: attributes."agent.name", 
                tool: attributes."tool.name", 
                duration: end_time - start_time,
                success: status.status_code == 'OK'
            }
        """)
        
        # Error detection pattern
        .query("[?status.status_code != 'OK']")
        .assert_spans(max_count=0)  # No errors allowed
        
        # Workflow completeness pattern
        .query("[?attributes.\"span.type\" == 'workflow']")
        .assert_spans(min_count=1)
        
        .completed_successfully())
```

## Business Logic Patterns

### Ordering and Dependencies

```python
def test_business_logic_patterns(self, travel_agent):
    """Test that business logic follows required patterns."""
    
    traces = self.assert_traces()
    
    (traces
        # Business rule: Always analyze before booking
        .assert_agent_called_before("request_analyzer", "booking_agent")
        
        # Business rule: Check availability before confirming
        .assert_agent_called_before("availability_checker", "booking_confirmer")
        
        # Business rule: Validate payment before processing
        .assert_tool_sequence("payment_processor", [
            "validate_payment_method",
            "authorize_payment",
            "process_payment"
        ])
        
        # Business rule: Audit trail completeness
        .assert_audit_pattern([
            "request_logged",
            "processing_started", 
            "booking_attempted",
            "result_confirmed",
            "audit_completed"
        ])
        
        .completed_successfully())
```

### Error Handling Patterns

```python
def test_error_handling_patterns(self, agent):
    """Test error handling and recovery patterns."""
    
    # Simulate error conditions
    agent.process_request_with_errors("Book unavailable flight")
    
    (self.assert_traces()
        # Error detection pattern
        .assert_error_handling_pattern("booking_agent", "retry_agent")
        
        # Retry pattern validation
        .assert_retry_pattern(
            failing_agent="booking_agent",
            max_retries=3,
            backoff_pattern="exponential"
        )
        
        # Graceful degradation pattern
        .assert_fallback_pattern(
            primary_agent="premium_booking_agent",
            fallback_agent="standard_booking_agent"
        )
        
        # Final completion despite errors
        .completed_successfully())
```

## Performance Patterns

### Timing and Resource Validation

```python
def test_performance_patterns(self, agent):
    """Test performance characteristics using patterns."""
    
    agent.process_large_workflow()
    
    (self.assert_traces()
        # Overall performance pattern
        .within_time_limit(10.0)
        
        # Critical path analysis pattern
        .assert_critical_path_duration(max_seconds=5.0)
        
        # Parallel efficiency pattern
        .assert_parallel_efficiency(
            parallel_agents=["worker1", "worker2", "worker3"],
            min_overlap_percentage=0.8
        )
        
        # Resource utilization pattern
        .assert_resource_usage_pattern(
            max_concurrent_llm_calls=5,
            max_memory_usage_mb=1000,
            max_api_calls_per_minute=100
        )
        
        # LLM call optimization pattern
        .assert_min_llm_calls(3)
        .assert_max_llm_calls(15)
        
        .completed_successfully())
```

### Scalability Patterns

```python
def test_scalability_patterns(self, agent):
    """Test that workflow scales appropriately."""
    
    # Test with increasing complexity
    for complexity in [1, 5, 10, 20]:
        agent.process_requests(count=complexity)
        
        (self.assert_traces()
            # Linear scaling pattern
            .assert_linear_scaling(
                input_size=complexity,
                max_time_per_item=0.5
            )
            
            # Bounded resource pattern
            .assert_bounded_resources(
                max_memory_growth_factor=2.0,
                max_time_growth_factor=1.5
            )
            
            .completed_successfully())
```

## Complete Flow Validation

### Comprehensive End-to-End Pattern

```python
class TestCompleteWorkflowPatterns(BaseAgentTest):
    """Complete workflow pattern validation suite."""
    
    def test_entire_flow_comprehensive_validation(self, travel_agent):
        """Test the complete flow using all available patterns."""
        
        request = "Plan comprehensive business trip to Mumbai with flights, hotel, car rental, and expense tracking"
        result = travel_agent.process_request(request)
        
        # THE COMPLETE FLOW PATTERN CHAIN
        traces = self.assert_traces()
        
        comprehensive_validation = (traces
            # === INITIATION PATTERNS ===
            .filter_by_name("workflow.start")
            .assert_spans(exactly=1)
            .contains_input("Mumbai")
            .contains_input("business trip")
            .contains_input("comprehensive")
            
            # === WORKFLOW STRUCTURE PATTERNS ===
            .reset_to_all_spans()
            .assert_agent_sequence([
                "request_processor",
                "travel_coordinator", 
                "resource_allocator"
            ])
            
            # === PARALLEL EXECUTION PATTERNS ===
            .assert_workflow_pattern("fan-out", [
                "travel_coordinator",
                "flight_booker", 
                "hotel_booker",
                "car_rental_agent"
            ])
            
            .assert_agents_called_in_parallel([
                "flight_booker",
                "hotel_booker", 
                "car_rental_agent"
            ], tolerance_ms=2000)
            
            # === TOOL INTERACTION PATTERNS ===
            .assert_tool_sequence("flight_booker", [
                "search_flights",
                "check_availability",
                "book_flight", 
                "confirm_booking"
            ])
            
            .assert_tool_sequence("hotel_booker", [
                "search_hotels",
                "check_rates",
                "book_hotel",
                "confirm_reservation"
            ])
            
            # === AGGREGATION PATTERNS ===
            .assert_workflow_pattern("fan-in", [
                "flight_booker",
                "hotel_booker",
                "car_rental_agent",
                "expense_tracker"
            ])
            
            # === BUSINESS LOGIC PATTERNS ===
            .assert_conditional_flow(
                condition_agent="travel_coordinator",
                condition_output_contains="business",
                then_agents=["expense_tracker", "receipt_manager"],
                else_agents=["vacation_optimizer"]
            )
            
            .assert_agent_called_before("availability_checker", "booking_confirmer")
            .assert_agent_called_after("expense_tracker", "booking_confirmer")
            
            # === DATA FLOW PATTERNS ===
            .assert_data_flow_pattern([
                ("request_processor", "travel_coordinator", "parsed_request"),
                ("travel_coordinator", "flight_booker", "flight_requirements"),
                ("flight_booker", "expense_tracker", "booking_details")
            ])
            
            # === OUTPUT VALIDATION PATTERNS ===
            .semantically_contains_output("flight confirmed", threshold=0.8)
            .semantically_contains_output("hotel booked", threshold=0.8)
            .semantically_contains_output("car rental arranged", threshold=0.7)
            .semantically_contains_output("expenses tracked", threshold=0.7)
            
            # === ERROR HANDLING PATTERNS ===
            .assert_no_critical_errors()
            .assert_graceful_error_handling()
            
            # === PERFORMANCE PATTERNS ===
            .within_time_limit(15.0)
            .assert_critical_path_duration(max_seconds=8.0)
            .assert_parallel_efficiency(
                parallel_agents=["flight_booker", "hotel_booker", "car_rental_agent"],
                min_overlap_percentage=0.7
            )
            
            # === RESOURCE PATTERNS ===
            .assert_resource_usage_pattern(
                max_concurrent_llm_calls=8,
                max_memory_usage_mb=2000,
                max_api_calls_total=50
            )
            
            # === AUDIT AND COMPLIANCE PATTERNS ===
            .assert_audit_trail_completeness([
                "request_received",
                "processing_initiated",
                "bookings_attempted", 
                "confirmations_sent",
                "expenses_calculated",
                "workflow_completed"
            ])
            
            # === FINAL VALIDATION PATTERNS ===
            .completed_successfully()
            .assert_workflow_integrity()
            .assert_all_agents_completed()
        )
        
        # Execute and debug the comprehensive validation
        comprehensive_validation.debug_execution_flow()
        comprehensive_validation.debug_entities()
        
        logger.info("✅ Comprehensive flow pattern validation completed successfully!")
        
        # Return for further analysis if needed
        return comprehensive_validation
```

## Advanced Query Patterns

### Custom JMESPath Patterns

```python
def test_custom_jmespath_patterns(self, agent):
    """Test using custom JMESPath queries for pattern validation."""
    
    agent.process_workflow()
    traces = self.assert_traces()
    
    # Complex entity extraction pattern
    workflow_summary = traces.query("""
        [?contains(attributes."span.type", 'agentic')].{
            workflow_type: attributes."span.type", 
            agent_name: attributes."entity.1.name", 
            operation: attributes."span.subtype",
            duration: (end_time - start_time) / 1000000000,
            tools_used: attributes."tool.name"
        } | [?workflow_type != null]
    """)
    
    # Performance analysis pattern
    performance_metrics = traces.query("""
        [].{
            name: name,
            duration_seconds: (end_time - start_time) / 1000000000,
            is_llm_call: contains(attributes."entity.1.type" || '', 'model.llm'),
            agent: attributes."agent.name" || attributes."entity.1.name"
        } | [?duration_seconds > 0]
    """)
    
    # Validate extracted patterns
    assert len(workflow_summary) > 0, "Should find workflow components"
    assert len(performance_metrics) > 0, "Should have performance data"
    
    # Chain pattern validations
    (traces
        .query(workflow_summary)  # Use extracted data
        .assert_custom_pattern_match(expected_agents=["agent1", "agent2"])
        .completed_successfully())
```

## Best Practices

### 1. Pattern Composition Strategy

```python
# Good: Compose patterns logically
def test_logical_pattern_composition(self, agent):
    (self.assert_traces()
        # Start with structural validation
        .assert_workflow_structure()
        
        # Then validate business logic
        .assert_business_rules()
        
        # Finally check performance
        .assert_performance_requirements()
        
        .completed_successfully())

# Avoid: Random pattern mixing without logic
```

### 2. Error Context Preservation

```python
# Good: Preserve context for debugging
def test_with_error_context(self, agent):
    try:
        (self.assert_traces()
            .assert_complex_pattern()
            .completed_successfully())
    except AssertionError as e:
        # Add debugging context before re-raising
        traces.debug_execution_flow()
        traces.debug_entities()
        logger.error(f"Pattern validation failed: {e}")
        raise
```

### 3. Modular Pattern Functions

```python
# Good: Create reusable pattern functions
def assert_booking_workflow_pattern(traces):
    """Reusable booking workflow pattern."""
    return (traces
        .assert_agent_sequence(["analyzer", "booker", "confirmer"])
        .assert_tool_usage_pattern("booker", ["search", "book", "confirm"])
        .within_time_limit(5.0))

def test_flight_booking(self, agent):
    assert_booking_workflow_pattern(self.assert_traces()).completed_successfully()

def test_hotel_booking(self, agent):  
    assert_booking_workflow_pattern(self.assert_traces()).completed_successfully()
```

### 4. Progressive Validation

```python
# Good: Build validation progressively
def test_progressive_validation(self, agent):
    traces = self.assert_traces()
    
    # Level 1: Basic structure
    basic_validation = traces.assert_basic_structure()
    
    # Level 2: Add business logic (depends on Level 1)
    business_validation = basic_validation.assert_business_logic()
    
    # Level 3: Add performance (depends on Level 2)
    complete_validation = business_validation.assert_performance()
    
    complete_validation.completed_successfully()
```

### 5. Pattern Documentation

```python
def test_documented_patterns(self, agent):
    """
    Test comprehensive workflow patterns.
    
    Pattern Flow:
    1. Request Processing → Travel Analysis
    2. Travel Analysis → [Flight Booking ‖ Hotel Booking] (parallel)
    3. [Flight Booking ‖ Hotel Booking] → Expense Tracking (fan-in)
    4. Expense Tracking → Confirmation
    
    Expected Timing:
    - Total workflow: < 10 seconds
    - Parallel booking: < 5 seconds overlap
    - Critical path: Request → Analysis → Confirmation < 6 seconds
    """
    
    (self.assert_traces()
        # Implement the documented pattern exactly
        .assert_agent_sequence(["request_processor", "travel_analyzer"])
        .assert_agents_called_in_parallel(["flight_booker", "hotel_booker"])
        .assert_workflow_pattern("fan-in", ["flight_booker", "hotel_booker", "expense_tracker"])
        .assert_agent_called_after("expense_tracker", "confirmation_sender")
        
        # Validate documented timing requirements
        .within_time_limit(10.0)
        .assert_parallel_overlap("flight_booker", "hotel_booker", min_seconds=3.0)
        .assert_critical_path_duration(max_seconds=6.0)
        
        .completed_successfully())
```

## Span Name vs Span Type Matching in Flow Patterns

### Understanding Pattern Matching Priority

The `assert_flow` method uses a prioritized matching strategy to differentiate between span names and span types. Understanding this behavior is crucial for writing effective flow patterns.

#### Matching Strategy Order

The flow validator uses this **order of precedence** for each pattern in your flow:

##### 1. **Exact Name Match** (Highest Priority)
```python
if event.name == step_pattern:
    return True
```
- If `step_pattern` exactly equals `event.name`, it matches
- Example: `"validate_user_data"` matches span with `name = "validate_user_data"`

##### 2. **Pattern/Wildcard Match**
```python
if "*" in step_pattern or "?" in step_pattern:
    pattern_regex = step_pattern.replace("*", ".*").replace("?", ".?")
    if re.match(pattern_regex, event.name):
        return True
```
- Supports wildcards against span names
- Example: `"validate_*"` matches `"validate_user_data"`

##### 3. **Span Type Match** (Lower Priority)
```python
if event.span_type == step_pattern:
    return True
```
- If no name match, tries to match against `span.type` attribute
- Example: `"http.process"` matches span with `attributes["span.type"] = "http.process"`

##### 4. **Partial Name Match** (Lowest Priority)
```python
if step_pattern.lower() in event.name.lower():
    return True
```
- Case-insensitive substring matching
- Example: `"user"` matches `"validate_user_data"`

### Practical Examples

#### Mixed Span Type and Name Matching

```python
traces.assert_flow("http.process -> validate_user_data -> calculate_user_profile_score")
```

Here's how each part matches:

1. **`"http.process"`**: 
   - Tries exact name match first → No span named exactly "http.process"
   - Tries wildcard match → No wildcards in pattern
   - **Tries span type match → ✅ Matches span with `span.type = "http.process"`**

2. **`"validate_user_data"`**:
   - **Tries exact name match → ✅ Matches span with `name = "validate_user_data"`**

3. **`"calculate_user_profile_score"`**:
   - **Tries exact name match → ✅ Matches span with `name = "calculate_user_profile_score"`**

#### Disambiguation Strategies

##### **Explicit Span Type Matching**
```python
# These will prioritize span type matching:
traces.assert_flow("http.process")  # Will match span.type
traces.assert_flow("agent.openai")  # Will match span.type
```

##### **Explicit Span Name Matching**  
```python
# These will prioritize exact name matching:
traces.assert_flow("validate_user_data")  # Will match span.name
traces.assert_flow("create_user")         # Will match span.name
```

##### **Mixed Matching (Recommended)**
```python
# This works because the patterns are unambiguous:
traces.assert_flow("http.process -> validate_user_data")
# - "http.process" only exists as a span.type (no span named "http.process")
# - "validate_user_data" only exists as a span.name (no span.type "validate_user_data")
```

#### Handling Potential Conflicts

If you had both a span named `"http.process"` AND a span with `span.type = "http.process"`, the **exact name match would win** due to the priority order.

In practice, conflicts are rare because:
- `"http.process"` typically only exists as a span type
- `"validate_user_data"` typically only exists as a span name
- The flow validator automatically picks the right matching strategy for each pattern

#### Advanced Pattern Examples

##### **Complex Mixed Patterns**
```python
# HTTP processing, then validation, then business logic
traces.assert_flow("http.process -> validate_* -> (calculate_* -> send_*) -> audit_*")
```

##### **Span Type Hierarchies**
```python
# Match different levels of abstraction
traces.assert_flow("http.process -> agent.openai -> tool.function_call")
```

##### **Partial Matching with Context**
```python
# Match spans containing certain words
traces.assert_flow("*user* -> *notification*")  # Any span containing "user" then "notification"
```

### Best Practices for Pattern Matching

1. **Use Descriptive Names**: Make span names and types clearly distinguishable
2. **Leverage Priority**: Rely on the natural matching priority for cleaner patterns
3. **Test Ambiguity**: Verify your patterns work with the actual span data
4. **Document Intent**: Add comments explaining complex pattern matching logic

## Conclusion

The Monocle Testing Framework's pattern chaining capabilities allow you to:

1. **Validate Complete Workflows** - Test entire trace flows from initiation to completion
2. **Combine Multiple Patterns** - Chain sequential, parallel, conditional, and custom patterns
3. **Ensure Business Logic** - Validate that workflows follow required business rules
4. **Monitor Performance** - Ensure workflows meet timing and resource requirements
5. **Debug Complex Flows** - Use comprehensive debugging and visualization tools

By chaining these patterns together, you can create comprehensive, maintainable tests that validate both the technical implementation and business logic of your AI agent workflows.