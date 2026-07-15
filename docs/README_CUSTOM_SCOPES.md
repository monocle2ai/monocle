# Custom Scope Validation for Monocle

Enhanced trace validation framework that supports fetching traces from Okahu cloud using **arbitrary custom scopes**, not just predefined session IDs.

## Overview

This feature enables flexible trace grouping and validation for:
- **Test isolation** - Tag traces with unique test IDs
- **User journey tracking** - Group all traces by user
- **CI/CD integration** - Tag traces by CI run ID
- **Any custom grouping** you need

## Quick Start

### 1. Generate Traces with Custom Scope

```python
from monocle_apptrace.utils import start_scope, stop_scope

# Create a custom scope
token = start_scope("test_id", "test_abc_123")

# Your code generates traces here
result = my_agent.run("query")

# Stop the scope
stop_scope(token)
```

### 2. Validate Traces by Custom Scope

```python
from monocle_test_tools.fluent_api import TraceAssertion

# Fetch traces by custom scope
asserter = TraceAssertion()
asserter.import_traces(
    trace_source="okahu",
    id="test_abc_123",
    fact_name="scope",
    scope_name="test_id",
    workflow_name="my_app"
)

# Run assertions
asserter.called_tool("search").has_output("result")
asserter.under_token_limit(5000)
```

## Key Concepts

### Trace ID
Unique identifier for one execution flow. Groups all spans from a single execution.
```
trace_id: "769d08fe96473a0551d338ad39fa1922"
```

### Scope (Fact)
A grouping mechanism across multiple traces:
- **scope_name** (fact_name): The category (e.g., "test_id", "user_id")
- **scope_id** (fact_id): The specific value (e.g., "test_123", "user_456")

```
Scope: test_id="test_123"
  ├── Trace: abc123 → Span 1, Span 2
  └── Trace: def456 → Span 3, Span 4
```

## Usage Examples

### Test Isolation

```python
from monocle_apptrace.utils import start_scope, stop_scope
from monocle_test_tools.fluent_api import TraceAssertion

def test_my_feature():
    # Generate traces with unique test ID
    test_id = f"test_{uuid.uuid4()}"
    token = start_scope("test_id", test_id)
    
    try:
        result = my_agent.run("test query")
    finally:
        stop_scope(token)
    
    # Validate traces from this test only
    asserter = TraceAssertion()
    asserter.import_traces(
        trace_source="okahu",
        id=test_id,
        fact_name="scope",
        scope_name="test_id",
        workflow_name="my_app"
    )
    
    asserter.called_tool("search").has_output("success")
```

### User Journey Tracking

```python
# In your application
def handle_request(user_id, request):
    token = start_scope("user_id", user_id)
    try:
        return process_request(request)
    finally:
        stop_scope(token)

# In validation
asserter = TraceAssertion()
asserter.import_traces(
    trace_source="okahu",
    id="user_456",
    fact_name="scope",
    scope_name="user_id",
    workflow_name="production"
)
```

### CI/CD Integration

```python
# In CI pipeline
ci_run_id = os.environ["CI_RUN_ID"]
token = start_scope("ci_run_id", ci_run_id)

try:
    run_integration_tests()
finally:
    stop_scope(token)

# Validate CI run
asserter = TraceAssertion()
asserter.import_traces(
    trace_source="okahu",
    id=ci_run_id,
    fact_name="scope",
    scope_name="ci_run_id",
    workflow_name="ci_tests"
)

asserter.under_token_limit(100000)
asserter.under_duration(60, units="seconds")
```

## API Reference

### import_traces()

```python
asserter.import_traces(
    trace_source: str,           # "okahu" or "file"
    id: str,                     # scope_id, session_id, or trace_id
    fact_name: str = "trace",    # "trace", "session", or "scope"
    scope_name: str = None,      # Required when fact_name="scope"
    workflow_name: str = None    # Required for okahu source
)
```

**Parameters:**
- `trace_source`: Source type - "okahu" or "file"
- `id`: The identifier to fetch (meaning depends on fact_name)
- `fact_name`: 
  - `"trace"` - Single trace lookup
  - `"session"` - Session lookup (uses agent_sessions)
  - `"scope"` - Custom scope lookup (requires scope_name)
- `scope_name`: Name of custom scope (required when fact_name="scope")
- `workflow_name`: Okahu workflow name (required for okahu source)

**Examples:**

```python
# By trace ID
asserter.import_traces("okahu", "abc123", "trace", workflow_name="app")

# By session (backward compatible)
asserter.import_traces("okahu", "session_123", "session", workflow_name="app")

# By custom scope (NEW)
asserter.import_traces(
    "okahu", "test_456", "scope", 
    scope_name="test_id", workflow_name="app"
)
```

### OkahuSpanLoader.load_by_scope()

```python
from monocle_test_tools.span_loader import OkahuSpanLoader

spans = OkahuSpanLoader.load_by_scope(
    workflow_name="my_app",
    scope_name="test_id",
    scope_id="test_123",
    endpoint=None,
    api_key=None,
    timeout=60
)
```

Returns `List[ReadableSpan]` for all traces matching the scope.

## Testing

### Run Integration Tests

```bash
pytest apptrace/tests/integration/test_cloud_trace_validation.py -v -s
```



## Environment Setup

Required environment variables:

```bash
export OKAHU_API_KEY="your_api_key"
export OKAHU_API_ENDPOINT="https://api.okahu.co"
```

Optional:
```bash
export MONOCLE_EXPORTER="okahu"
```

## Files Modified

### Core Implementation
- `test_tools/src/monocle_test_tools/span_loader.py`
  - Added `OkahuSpanLoader.load_by_scope()` method
- `test_tools/src/monocle_test_tools/fluent_api.py`
  - Enhanced `import_traces()` with scope_name parameter

### Documentation & Examples
- `apptrace/tests/integration/test_cloud_trace_validation.py` - Integration tests for cloud trace validation

## Troubleshooting

### 401 Authentication Error
```
Solution: Update OKAHU_API_KEY
export OKAHU_API_KEY="your_valid_key"
```

### 404 Not Found
```
Problem: Trace/scope doesn't exist in Okahu
Solutions:
- Verify traces were exported
- Check workflow_name is correct
- Check scope_name matches start_scope() call
- Wait a few seconds for Okahu to process
```

### Missing Endpoint
```
Solution: Set OKAHU_API_ENDPOINT
export OKAHU_API_ENDPOINT="https://api.okahu.co"
```

### ValueError: 'scope_name' required
```
Problem: Using fact_name="scope" without scope_name
Solution: Add scope_name parameter

asserter.import_traces(
    "okahu", "test_123", "scope",
    scope_name="test_id",  # ← Add this
    workflow_name="app"
)
```

## Backward Compatibility

All existing code continues to work:

```python
# Session-based (still works)
asserter.import_traces("okahu", "session_123", "session", workflow_name="app")

# Trace ID (still works)
asserter.import_traces("okahu", "abc123", "trace", workflow_name="app")

# File-based (still works)
asserter.import_traces("file", "abc123")
```

## REST API Endpoints

The implementation uses these Okahu endpoints:

1. **Get trace IDs for a scope:**
   ```
   GET /api/v1/workflows/{workflow}/traces?duration_fact={scope_name}&fact_ids={scope_id}
   ```

2. **Get spans for a trace:**
   ```
   GET /api/v1/workflows/{workflow}/traces/{trace_id}/spans?filter_fact={scope_name}&filter_fact_id={scope_id}
   ```

## Common Scope Names

- `agent_sessions` - Built-in for agent sessions
- `test_id` - Custom test runs
- `user_id` - User-specific traces
- `ci_run_id` - CI/CD pipeline runs
- `request_id` - Request tracking
- Any custom name you define!

## Support

- **Examples**: [examples/custom_scope_e2e_example.py](examples/custom_scope_e2e_example.py)
- **Ind-to-End Example**: [examples/custom_scope_e2e_example.py](examples/custom_scope_e2e_example.py)
- **Integr
## Requirements

- Python 3.8+
- Monocle apptrace and test_tools packages
- Valid OKAHU_API_KEY
- Access to Okahu instance

## License

See [LICENSE](LICENSE)
