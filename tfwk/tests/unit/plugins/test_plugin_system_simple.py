"""
Simple test script to demonstrate the TraceAssertions plugin system.
"""
import sys
from datetime import datetime
from unittest.mock import MagicMock

from monocle_tfwk.assertions import TraceAssertions

# Import after mocking
from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin

# Mock the problematic imports to avoid dependency issues
sys.modules['monocle_apptrace.exporters.file_exporter'] = MagicMock()
sys.modules['monocle_apptrace.instrumentation'] = MagicMock()


def create_mock_span(name="test_span", attributes=None):
    """Create a mock ReadableSpan for testing."""
    span = MagicMock()
    span.name = name
    span.attributes = attributes or {}
    span.start_time = int(datetime.now().timestamp() * 1_000_000_000)
    span.end_time = span.start_time + 100_000_000  # 0.1 seconds later
    
    # Mock the status
    status = MagicMock()
    status.status_code = MagicMock()
    status.status_code.name = "OK"
    span.status = status
    
    return span


def main():
    """Demonstrate the plugin system."""
    print("=== TraceAssertions Plugin System Demo ===\n")
    
    # Create mock spans
    spans = [
        create_mock_span("agent_call", {"agent.name": "test_agent", "llm.usage.total_cost": "0.05"}),
        create_mock_span("llm_inference", {"span.type": "inference", "cost": "0.02"})
    ]
    
    # Define a custom plugin
    @plugin
    class TestCustomPlugin(TraceAssertionsPlugin):
        """A simple test plugin."""
        
        @classmethod
        def get_plugin_name(cls) -> str:
            return "test_custom"
        
        def assert_test_condition(self, value: str) -> 'TraceAssertions':
            """Test assertion method."""
            print(f"   Custom assertion called with value: {value}")
            return self
        
        def contains_test_data(self, data: str) -> 'TraceAssertions':
            """Test that spans contain specific test data."""
            found = False
            for span in self._current_spans:
                if hasattr(span, 'name') and data in span.name:
                    found = True
                    break
            
            if not found:
                print(f"   Warning: Test data '{data}' not found in any spans")
            return self
    
    # Create TraceAssertions instance
    assertions = TraceAssertions(spans)
    
    print("1. Testing built-in methods:")
    try:
        assertions.assert_spans(count=2)
        print("   ✓ Built-in assert_spans works")
    except Exception as e:
        print(f"   ✗ Built-in method failed: {e}")
    
    print("\n2. Testing plugin methods:")
    
    # Test custom plugin methods
    try:
        assertions.assert_test_condition("hello")
        print("   ✓ Custom plugin method works")
    except Exception as e:
        print(f"   ✗ Custom plugin method failed: {e}")
    
    try:
        assertions.contains_test_data("agent")
        print("   ✓ Custom plugin assertion works")
    except Exception as e:
        print(f"   ✗ Custom plugin assertion failed: {e}")
    
    print("\n3. Testing method chaining with plugins:")
    try:
        (assertions
         .assert_spans(min_count=1)
         .assert_test_condition("chained")
         .contains_test_data("test"))
        print("   ✓ Method chaining with plugins works")
    except Exception as e:
        print(f"   ✗ Method chaining failed: {e}")
    
    print("\n4. Listing registered plugins:")
    plugins = TraceAssertions.list_plugins()
    for name, plugin_class in plugins.items():
        methods = plugin_class.get_assertion_methods()
        print(f"   Plugin '{name}': {list(methods.keys())}")
    
    print("\n5. Testing dynamic method availability:")
    # Check if plugin methods are available on the instance
    available_methods = [method for method in dir(assertions) 
                        if not method.startswith('_') and callable(getattr(assertions, method))]
    plugin_methods = ['assert_test_condition', 'contains_test_data']
    
    for method in plugin_methods:
        if method in available_methods:
            print(f"   ✓ Plugin method '{method}' is available")
        else:
            print(f"   ✗ Plugin method '{method}' is not available")
    
    print("\n6. Testing plugin registration/unregistration:")
    initial_count = len(TraceAssertions.list_plugins())
    
    # Register another plugin
    @plugin
    class AnotherTestPlugin(TraceAssertionsPlugin):
        @classmethod
        def get_plugin_name(cls) -> str:
            return "another_test"
        
        def another_method(self) -> 'TraceAssertions':
            print("   Another plugin method called")
            return self
    
    # Create new instance to get the new plugin methods
    new_assertions = TraceAssertions(spans)
    final_count = len(TraceAssertions.list_plugins())
    
    print(f"   Initial plugins: {initial_count}, Final plugins: {final_count}")
    
    if hasattr(new_assertions, 'another_method'):
        new_assertions.another_method()
        print("   ✓ Dynamic plugin registration works")
    else:
        print("   ✗ Dynamic plugin registration failed")
    
    print("\n7. Testing plugin extensibility example:")
    
    # Define a performance monitoring plugin
    @plugin
    class PerformancePlugin(TraceAssertionsPlugin):
        @classmethod
        def get_plugin_name(cls) -> str:
            return "performance_test"
        
        def assert_fast_execution(self, max_seconds: float = 1.0) -> 'TraceAssertions':
            """Assert all spans execute within time limit."""
            slow_spans = []
            for span in self._current_spans:
                duration = (span.end_time - span.start_time) / 1_000_000_000
                if duration > max_seconds:
                    slow_spans.append((span.name, duration))
            
            if slow_spans:
                print(f"   Warning: Found slow spans: {slow_spans}")
            else:
                print(f"   ✓ All spans executed within {max_seconds}s")
            return self
    
    # Test the new plugin
    perf_assertions = TraceAssertions(spans)
    perf_assertions.assert_fast_execution(0.5)
    
    print("\n=== Demo Complete ===")
    print("\nThe TraceAssertions class is now fully extensible!")
    print("Users can create custom plugins to add domain-specific assertions.")


if __name__ == "__main__":
    main()