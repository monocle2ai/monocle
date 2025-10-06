# Monocle Integration Testing Guide

This guide provides comprehensive instructions for writing reliable integration tests for Monocle-instrumented applications.

## Overview

When writing integration tests for your Monocle-instrumented applications, proper setup and cleanup are crucial to avoid test interference and global state leakage. The key challenge is that `setup_monocle_telemetry` creates global state that persists across tests, requiring careful management.

## Test Setup with Fixture and Exporter Access

Use pytest fixtures to set up Monocle telemetry and capture spans for validation:

```python
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

@pytest.fixture(scope="function")
def setup():
    custom_exporter = CustomConsoleSpanExporter()
    instrumentor = None
    try:
        # Setup Monocle telemetry with custom exporter for span capture
        instrumentor = setup_monocle_telemetry(
            workflow_name="test_app",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
            wrapper_methods=[]
        )
        # Yield the exporter so tests can access captured spans
        yield custom_exporter
    finally:
        # CRITICAL: Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def test_my_functionality(setup):
    custom_exporter = setup
    
    # Your test code here
    # ... execute traced operations ...
    
    # Access captured spans for validation
    captured_spans = custom_exporter.get_captured_spans()
    assert len(captured_spans) > 0
```

## Why `instrumentor.uninstrument()` is Essential

The `setup_monocle_telemetry` function creates **global state** that persists across tests, even after the fixture terminates. Without proper cleanup, subsequent tests will be affected by leftover instrumentation state, leading to:

- Unexpected spans from previous tests
- Interference between test cases  
- Unpredictable test behavior
- False positives/negatives in assertions

### The Global State Problem

**The global state problem occurs because:**

1. `set_tracer_provider(TracerProvider(...))` - Sets a **GLOBAL** tracer provider
2. `trace.set_tracer_provider(...)` - Sets global OpenTelemetry state
3. `instrumentor.instrument(...)` - Applies global instrumentation to libraries/frameworks

**Solution:** Always call `instrumentor.uninstrument()` in the fixture's `finally` block to clean up the global instrumentation state.

## Understanding Fixture Scope and State Management

### Function-scoped Fixtures (Recommended for most tests)

```python
@pytest.fixture(scope="function") 
def setup():
    # Fresh setup for each test
    # Prevents cross-test contamination
```

- **Use when:** Each test needs isolated instrumentation state
- **Benefits:** Complete isolation, predictable test behavior
- **Drawbacks:** Slight performance overhead from repeated setup

### Module-scoped Fixtures (Use with caution)

```python
@pytest.fixture(scope="module")
def setup():
    # Shared setup across all tests in the module
    # Still need cleanup to avoid affecting other modules
```

- **Use when:** Tests in a module can safely share instrumentation state
- **Benefits:** Better performance for test suites
- **Requirements:** All tests must be compatible with shared state
- **Still need:** `uninstrument()` cleanup to avoid affecting other test modules

## Complete Example with Environment Cleanup

```python
import os
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

@pytest.fixture(scope="function")
def setup():
    # Save original environment state
    original_env_var = os.environ.get("SOME_CONFIG_PATH")
    instrumentor = None
    
    try:
        # Setup custom exporter to capture spans
        custom_exporter = CustomConsoleSpanExporter()
        
        # Set test-specific environment if needed
        os.environ["SOME_CONFIG_PATH"] = "/path/to/test/config"
        
        # Setup Monocle telemetry
        instrumentor = setup_monocle_telemetry(
            workflow_name="integration_test",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
            wrapper_methods=[]
        )
        
        yield custom_exporter
        
    finally:
        # CRITICAL: Uninstrument to clean up global state
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
            
        # Restore original environment state
        if original_env_var is not None:
            os.environ["SOME_CONFIG_PATH"] = original_env_var
        elif "SOME_CONFIG_PATH" in os.environ:
            del os.environ["SOME_CONFIG_PATH"]

def test_trace_generation(setup):
    """Test that demonstrates span capture and validation"""
    custom_exporter = setup
    
    # Execute your traced operations
    # ... your application code here ...
    
    # Validate captured spans
    spans = custom_exporter.get_captured_spans()
    assert len(spans) > 0, "Expected spans to be captured"
    
    # Reset exporter for next assertion if needed
    custom_exporter.reset()
```

## Working with Custom Exporter

The `CustomConsoleSpanExporter` is a specialized exporter that captures spans for test validation:

```python
# Access captured spans
captured_spans = custom_exporter.get_captured_spans()

# Validate span properties
for span in captured_spans:
    assert span.name is not None
    assert span.context.trace_id is not None

# Reset captured spans between tests or assertions
custom_exporter.reset()
```

## Common Integration Test Patterns

### Testing Specific Framework Integration

```python
def test_langchain_integration(setup):
    custom_exporter = setup
    
    # Execute LangChain operations
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI()
    result = llm.invoke("Test message")
    
    # Validate Monocle captured the LangChain spans
    spans = custom_exporter.get_captured_spans()
    langchain_spans = [s for s in spans if "langchain" in s.name.lower()]
    assert len(langchain_spans) > 0
```

### Testing Custom Wrapper Methods

```python
@pytest.fixture(scope="function")
def setup_with_custom_wrapper():
    custom_exporter = CustomConsoleSpanExporter()
    instrumentor = None
    
    try:
        custom_wrapper_methods = [
            # Your custom wrapper method configurations
        ]
        
        instrumentor = setup_monocle_telemetry(
            workflow_name="custom_wrapper_test",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
            wrapper_methods=custom_wrapper_methods
        )
        
        yield custom_exporter
        
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
```

### Testing with Multiple Span Processors

```python
def setup_with_multiple_processors():
    custom_exporter = CustomConsoleSpanExporter()
    file_exporter = FileSpanExporter("test_traces.json")
    instrumentor = None
    
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="multi_processor_test",
            span_processors=[
                SimpleSpanProcessor(custom_exporter),
                BatchSpanProcessor(file_exporter)
            ]
        )
        
        yield custom_exporter
        
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
```

## Key Best Practices

1. **Always use try/finally**: Ensure cleanup happens even if test setup fails
2. **Check instrumentor state**: Verify `instrumentor.is_instrumented_by_opentelemetry` before calling `uninstrument()`
3. **Environment cleanup**: Restore original environment variables to prevent side effects
4. **Exporter access**: Use `yield` to provide test access to the custom exporter for span validation
5. **State isolation**: Prefer function-scoped fixtures unless you specifically need shared state
6. **Reset exporters**: Call `custom_exporter.reset()` between assertions in the same test if needed
7. **Validate spans**: Always check that expected spans are captured and have correct properties
8. **Handle async code**: Use appropriate async fixtures and test methods for async operations

## Debugging Test Issues

### Common Problems and Solutions

**Problem**: Tests pass individually but fail when run together
- **Cause**: Global state leakage between tests
- **Solution**: Ensure all fixtures call `instrumentor.uninstrument()`

**Problem**: Unexpected spans appear in test results
- **Cause**: Previous test's instrumentation still active
- **Solution**: Verify fixture cleanup and consider function-scoped fixtures

**Problem**: No spans captured despite instrumentation
- **Cause**: Exporter not properly configured or framework not instrumented
- **Solution**: Check wrapper methods and verify framework support

**Problem**: Environment variables affecting tests
- **Cause**: Tests modifying global environment without cleanup
- **Solution**: Save and restore environment state in fixtures

## Environment Variable Management with conftest.py

### The Environment Variable Problem

One of the most common sources of test failures in integration tests is **environment variable contamination**. When one test sets environment variables (like API keys, configuration paths, or feature flags), these changes persist globally and can cause subsequent tests to fail or behave unexpectedly.

**Common scenarios that cause test interference:**

1. **API Keys**: Test A sets `OPENAI_API_KEY="test_key"`, Test B expects the real API key
2. **Configuration Paths**: Test A sets `SCOPE_CONFIG_PATH="/test/path"`, Test B uses default path
3. **Feature Flags**: Test A enables `MONOCLE_DEBUG=true`, affecting all subsequent tests
4. **Model Settings**: Test A sets `OPENAI_MODEL="gpt-3.5"`, Test B expects `gpt-4`

### Solution: Using conftest.py Utilities

The `apptrace/tests/config/conftest.py` file provides utilities to safely manage environment variables in tests:

```python
"""
Common pytest fixtures for all test modules.
"""
import os
from contextlib import contextmanager
import pytest

@pytest.fixture
def preserve_env():
    """Fixture to preserve and restore environment variables."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@contextmanager
def temporary_env_var(key, value):
    """Context manager to temporarily set an environment variable."""
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is not None:
            os.environ[key] = original_value
        elif key in os.environ:
            del os.environ[key]

@contextmanager
def temporary_env_vars(**env_vars):
    """Context manager to temporarily set multiple environment variables."""
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is not None:
                os.environ[key] = original_value
            elif key in os.environ:
                del os.environ[key]
```

### Usage Patterns

#### 1. Testing with Temporary Environment Variables

```python
from config.conftest import temporary_env_var

def test_with_invalid_api_key(setup):
    """Test error handling with invalid API key"""
    with temporary_env_var("OPENAI_API_KEY", "INVALID_KEY"):
        # Test code that should handle invalid API key
        # Environment variable is automatically restored after the block
        pass
    # Original OPENAI_API_KEY value is restored here
```

#### 2. Testing with Multiple Environment Variables

```python
from config.conftest import temporary_env_vars

def test_with_multiple_env_vars(setup):
    """Test with multiple temporary environment variables"""
    with temporary_env_vars(
        OPENAI_API_KEY="test_key",
        OPENAI_MODEL="gpt-3.5-turbo",
        MONOCLE_DEBUG="true"
    ):
        # Test code with temporary environment
        pass
    # All original values restored automatically
```

#### 3. Global Environment Preservation with `preserve_env()`

The `preserve_env()` fixture is ideal when your test needs to make extensive environment modifications or when you need to backup the entire environment state:

```python
def test_with_env_preservation(preserve_env, setup):
    """Test that modifies environment but ensures cleanup"""
    # This test can modify os.environ directly without worrying about cleanup
    os.environ["SOME_CONFIG"] = "test_value"
    os.environ["ANOTHER_CONFIG"] = "another_value"
    os.environ["THIRD_CONFIG"] = "third_value"
    
    # Test code that relies on these environment variables
    
    # No manual cleanup needed - preserve_env fixture handles complete restoration
```

**When to use `preserve_env()`:**

- **Extensive environment modifications**: When your test needs to set/modify many environment variables
- **Complex test scenarios**: When environment changes are scattered throughout the test
- **Unknown environment impact**: When you're not sure which environment variables might be affected
- **Legacy test migration**: When converting existing tests that manually manage environment state
- **Environment backup safety**: When you want complete assurance that the original environment is restored

**How `preserve_env()` works:**

1. **Backup**: Creates a complete copy of `os.environ` at fixture start
2. **Test execution**: Your test can freely modify environment variables
3. **Complete restoration**: Clears current environment and restores the original state
4. **Exception safety**: Restoration happens even if the test fails

```python
# Example: Testing configuration loading with multiple environment variables
def test_config_loading_with_multiple_vars(preserve_env, setup):
    """Test configuration loading with various environment combinations"""
    
    # Set up a complex environment scenario
    os.environ["MONOCLE_DEBUG"] = "true"
    os.environ["MONOCLE_EXPORTERS"] = "console,file"
    os.environ["OPENAI_API_KEY"] = "test-key-123"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
    os.environ["SCOPE_CONFIG_PATH"] = "/tmp/test-scope-config.json"
    
    # Test code that uses these environment variables
    config = load_monocle_configuration()
    assert config.debug_enabled is True
    assert "console" in config.exporters
    
    # Environment automatically restored to original state after test
```

### Real-World Examples from the Codebase

#### Example 1: Testing Invalid API Keys

```python
# From test_adk_single_agent.py
@pytest.mark.asyncio
async def test_invalid_api_key_error_code_in_span(setup):
    """Test that passing an invalid API key results in error_code in the span."""
    with temporary_env_var("GOOGLE_API_KEY", "INVALID_API_KEY"):
        try:
            test_message = "What is the current weather in New York?"
            await run_agent(test_message)
        except Exception:
            spans = setup.get_finished_spans()
            # Validate error spans
```

#### Example 2: Configuration Path Management

```python
# From test_scopes.py - Manual approach (now you can use conftest utilities)
@pytest.fixture(scope="function")
def setup():
    # Save original environment variable value
    original_scope_config = os.environ.get(SCOPE_CONFIG_PATH)
    try:
        os.environ[SCOPE_CONFIG_PATH] = "/path/to/test/config"
        # Setup code...
        yield custom_exporter
    finally:
        # Restore original environment variable value
        if original_scope_config is not None:
            os.environ[SCOPE_CONFIG_PATH] = original_scope_config
        elif SCOPE_CONFIG_PATH in os.environ:
            del os.environ[SCOPE_CONFIG_PATH]

# Better approach using conftest utilities:
def test_with_scope_config(setup):
    with temporary_env_var(SCOPE_CONFIG_PATH, "/path/to/test/config"):
        # Test code here
        pass
```

### Why This Approach is Better

1. **Automatic Cleanup**: Environment variables are automatically restored
2. **Exception Safety**: Cleanup happens even if test fails
3. **Isolation**: Each test gets a clean environment state
4. **Readability**: Clear intent about temporary environment changes
5. **Reusability**: Common patterns available across all tests

### Integration with Monocle Test Fixtures

Combine conftest utilities with Monocle fixtures for complete isolation:

```python
from config.conftest import temporary_env_vars

@pytest.fixture(scope="function")
def setup_with_test_env():
    instrumentor = None
    try:
        with temporary_env_vars(
            MONOCLE_DEBUG="true",
            SCOPE_CONFIG_PATH="/test/config"
        ):
            custom_exporter = CustomConsoleSpanExporter()
            instrumentor = setup_monocle_telemetry(
                workflow_name="isolated_test",
                span_processors=[SimpleSpanProcessor(custom_exporter)]
            )
            yield custom_exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        # Environment variables automatically restored by context managers
```

### Choosing the Right Environment Management Approach

| Scenario | Use | Why |
|----------|-----|-----|
| Setting 1-2 specific variables | `temporary_env_var()` | Precise control, clear intent |
| Setting multiple related variables | `temporary_env_vars()` | Clean syntax, atomic operation |
| Extensive environment modifications | `preserve_env()` fixture | Complete backup/restore |
| Complex test with scattered env changes | `preserve_env()` fixture | Safety net for unknown changes |
| Testing environment edge cases | `preserve_env()` fixture | Allows unrestricted modification |

### Best Practices for Environment Variable Management

1. **Use temporary_env_var for single variables** that need to be changed temporarily
2. **Use temporary_env_vars for multiple related variables** that should be changed together  
3. **Use preserve_env fixture for tests** that make extensive environment modifications or need complete environment backup
4. **Always prefer context managers over manual management** when you know which variables to change
5. **Use preserve_env as a safety net** when you're unsure about environment impact
6. **Document environment dependencies** in test docstrings
7. **Test both valid and invalid environment states** to ensure robustness
8. **Combine approaches when needed** - you can use `preserve_env` with `temporary_env_var` for extra safety

## Real-World Examples

Check the existing integration tests in this repository for practical examples:

- `test_scopes.py` - Demonstrates scope management and environment cleanup
- `test_langchain_chat_sample.py` - Shows module-scoped fixture usage
- `test_openai_*_sample.py` - Examples of different framework integrations
- `test_adk_single_agent.py` - Uses `temporary_env_var` for API key testing
- `test_langgraph_multi_agent.py` - Environment variable isolation patterns
- `config/conftest.py` - Provides reusable environment management utilities

Following these patterns ensures your integration tests are reliable, isolated, and don't interfere with each other.