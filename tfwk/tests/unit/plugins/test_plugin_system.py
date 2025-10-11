"""
Test script to demonstrate the TraceAssertions plugin system.
"""

import time

from monocle_tfwk.assertions import TraceAssertions

# Import plugins to register them (the classes are used via the @plugin decorator)
from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider


def create_mock_spans():
    """Create some mock spans for testing."""
    # Set up OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    spans = []
    
    # Create a mock span with some attributes
    with tracer.start_as_current_span("test_agent") as span:
        span.set_attribute("agent.name", "test_agent")
        span.set_attribute("llm.usage.total_cost", "0.05")
        span.set_attribute("llm.usage.total_tokens", "1000")
        time.sleep(0.1)  # Simulate some work
        spans.append(span)
    
    with tracer.start_as_current_span("another_span") as span:
        span.set_attribute("span.type", "inference")
        span.set_attribute("cost", "0.02")
        time.sleep(0.05)
        spans.append(span)
    
    return spans


@plugin
class TestCustomPlugin(TraceAssertionsPlugin):
    """A simple test plugin."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "test_custom"
    
    def assert_test_condition(self, value: str) -> 'TraceAssertions':
        """Test assertion method."""
        print(f"Custom assertion called with value: {value}")
        return self
    
    def contains_test_data(self, data: str) -> 'TraceAssertions':
        """Test that spans contain specific test data."""
        found = False
        for span in self._current_spans:
            if hasattr(span, 'name') and data in span.name:
                found = True
                break
        
        assert found, f"Test data '{data}' not found in any spans"
        return self


def main():
    """Demonstrate the plugin system."""
    print("=== TraceAssertions Plugin System Demo ===\n")
    
    # Create mock spans
    spans = create_mock_spans()
    
    # Create TraceAssertions instance
    assertions = TraceAssertions(spans)
    
    print("1. Testing built-in methods:")
    try:
        assertions.assert_spans(count=2)
        print("   ✓ Built-in assert_spans works")
    except Exception as e:
        print(f"   ✗ Built-in method failed: {e}")
    
    print("\n2. Testing plugin methods:")
    
    # Test performance plugin methods
    try:
        assertions.assert_total_duration_under(1.0)
        print("   ✓ Performance plugin method works")
    except Exception as e:
        print(f"   ✗ Performance plugin method failed: {e}")
    
    # Test cost plugin methods  
    try:
        assertions.assert_total_cost_under(1.0)
        print("   ✓ Cost plugin method works")
    except Exception as e:
        print(f"   ✗ Cost plugin method failed: {e}")
    
    # Test custom plugin methods
    try:
        assertions.assert_test_condition("hello")
        print("   ✓ Custom plugin method works")
    except Exception as e:
        print(f"   ✗ Custom plugin method failed: {e}")
    
    try:
        assertions.contains_test_data("test")
        print("   ✓ Custom plugin assertion works")
    except Exception as e:
        print(f"   ✗ Custom plugin assertion failed: {e}")
    
    print("\n3. Testing method chaining with plugins:")
    try:
        (assertions
         .assert_spans(min_count=1)
         .assert_total_duration_under(2.0)
         .assert_total_cost_under(1.0)
         .assert_test_condition("chained"))
        print("   ✓ Method chaining with plugins works")
    except Exception as e:
        print(f"   ✗ Method chaining failed: {e}")
    
    print("\n4. Listing registered plugins:")
    plugins = TraceAssertions.list_plugins()
    for name, plugin_class in plugins.items():
        methods = plugin_class.get_assertion_methods()
        print(f"   Plugin '{name}': {list(methods.keys())}")
    
    print("\n5. Testing plugin registration/unregistration:")
    initial_count = len(TraceAssertions.list_plugins())
    
    # Register another plugin
    @plugin
    class AnotherTestPlugin(TraceAssertionsPlugin):
        @classmethod
        def get_plugin_name(cls) -> str:
            return "another_test"
        
        def another_method(self) -> 'TraceAssertions':
            return self
    
    # Create new instance to get the new plugin methods
    new_assertions = TraceAssertions(spans)
    final_count = len(TraceAssertions.list_plugins())
    
    print(f"   Initial plugins: {initial_count}, Final plugins: {final_count}")
    
    if hasattr(new_assertions, 'another_method'):
        print("   ✓ Dynamic plugin registration works")
    else:
        print("   ✗ Dynamic plugin registration failed")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()