# Unified Flow Validation API

## Overview

We've simplified the Monocle flow validation API from multiple specific methods to a single, unified `assert_flow()` method that accepts flow patterns as strings.

## The Problem with the Old API

**Before (Multiple Methods):**
```python
# Different methods for different patterns - confusing and verbose
traces.assert_agent_sequence(["agent1", "agent2", "agent3"])
traces.assert_agents_called_in_parallel(["worker1", "worker2"])  
traces.assert_workflow_pattern("fan-out", ["supervisor", "worker1", "worker2"])
traces.assert_agent_called_before("agent1", "agent2")
traces.assert_conditional_flow("condition_agent", "condition", ["then_agents"], ["else_agents"])
```

## The Solution: Unified `assert_flow()` Method

**After (Single Method):**
```python
# Single method with intuitive pattern strings
traces.assert_flow("agent1 -> agent2 -> agent3")
traces.assert_flow("worker1 || worker2")
traces.assert_flow("supervisor -> (worker1 || worker2)")
traces.assert_flow("agent.reasoning -> (tool_use || (knowledge_lookup -> retrieval)) -> result_aggregation")
```

## Flow Pattern Syntax

| Operator | Meaning | Example |
|----------|---------|---------|
| `->` or `→` | Sequential (A must complete before B starts) | `"agent1 -> agent2"` |
| `\|\|` | Parallel (A and B execute with overlapping time) | `"worker1 \|\| worker2"` |
| `()` | Grouping for complex expressions | `"(A \|\| B) -> C"` |
| `\|` | Choice (either A or B, but not both) | `"toolA \| toolB"` |
| `?` | Optional (may or may not occur) | `"preprocessing?"` |
| `*` | Wildcard (matches any span containing pattern) | `"agent.*"` |
| `+` | One or more (repeats at least once) | `"retry+"` |

## Real-World Examples

### Simple Sequential Flow
```python
# RAG Pipeline
traces.assert_flow("embedding -> retrieval -> inference -> response")

# Agent Workflow  
traces.assert_flow("planning -> action -> reflection")
```

### Parallel Processing
```python
# Concurrent workers
traces.assert_flow("data_loader || cache_warmer || health_checker")

# Parallel inference
traces.assert_flow("model_a || model_b || model_c")
```

### Complex Nested Patterns
```python
# Conditional branching with nested sequences
traces.assert_flow("agent.reasoning -> (tool_use || (knowledge_lookup -> retrieval)) -> result_aggregation -> final_response")

# Fan-out then fan-in
traces.assert_flow("coordinator -> (worker1 || worker2 || worker3) -> aggregator -> finalizer")

# Optional preprocessing and postprocessing  
traces.assert_flow("preprocessing? -> inference -> postprocessing?")
```

### Error Handling Flows
```python
# Retry patterns
traces.assert_flow("primary_service -> (success | (retry -> backup_service))")

# Fallback chains
traces.assert_flow("fast_cache || (slow_db -> cache_update)")
```

## Benefits

1. **Unified API** - Single method instead of 6+ different methods
2. **Intuitive Syntax** - Pattern strings are self-documenting  
3. **Flexible** - Can express complex nested patterns easily
4. **Readable** - Flow patterns match natural language description
5. **Extensible** - Easy to add new operators and patterns
6. **Backward Compatible** - Legacy methods still work (deprecated)

## Migration Guide

### Old → New

```python
# Sequential
OLD: traces.assert_agent_sequence(["A", "B", "C"])
NEW: traces.assert_flow("A -> B -> C")

# Parallel  
OLD: traces.assert_agents_called_in_parallel(["X", "Y"])
NEW: traces.assert_flow("X || Y")

# Fan-out
OLD: traces.assert_workflow_pattern("fan-out", ["supervisor", "w1", "w2"])  
NEW: traces.assert_flow("supervisor -> (w1 || w2)")

# Fan-in
OLD: traces.assert_workflow_pattern("fan-in", ["w1", "w2", "aggregator"])
NEW: traces.assert_flow("(w1 || w2) -> aggregator")

# Before/After
OLD: traces.assert_agent_called_before("A", "B")
NEW: traces.assert_flow("A -> B")
```

## Implementation Details

The `assert_flow()` method:

1. **Parses** the pattern string into a validation structure
2. **Validates** timing relationships between agents/steps
3. **Supports** nested expressions with proper precedence  
4. **Provides** clear error messages when patterns fail
5. **Handles** optional tolerance parameters for parallel validation

```python
def assert_flow(self, pattern: str, tolerance_ms: int = 1000) -> 'TraceAssertions':
    """
    Assert that the execution flow matches the given pattern.
    
    Args:
        pattern: Flow pattern string (e.g., "A -> B || C")
        tolerance_ms: Time tolerance for parallel execution validation
    
    Returns:
        Self for method chaining
    
    Raises:
        AssertionError: If the flow doesn't match the pattern
    """
```

## Error Messages

The new API provides clear, actionable error messages:

```
AssertionError: Flow pattern validation failed: agent1 -> agent2 -> agent3
Error: Agent execution sequence mismatch.
Expected: agent1 → agent2 → agent3  
Actual: agent1 → agent3 → agent2
Agent 'agent2' started at 1635123456789, 'agent3' started at 1635123456123
```

## Conclusion

The unified `assert_flow()` API makes flow validation:
- **Simpler** to use (one method vs many)
- **More expressive** (complex patterns possible)  
- **Easier to read** (self-documenting patterns)
- **More maintainable** (consistent interface)

This change significantly improves the developer experience while maintaining full backward compatibility.