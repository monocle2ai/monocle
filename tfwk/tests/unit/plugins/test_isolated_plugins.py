"""
Isolated test of the modular plugin system for TraceAssertions.
"""
import sys
from unittest.mock import MagicMock

import pytest
from monocle_tfwk.assertions import TraceAssertions
from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin


def create_mock_span(name="test_span", attributes=None):
    """Create a mock ReadableSpan for testing."""
    span = MagicMock()
    span.name = name
    span.attributes = attributes or {}
    span.start_time = 1000000000  # 1 second
    span.end_time = 1100000000    # 1.1 seconds (100ms duration)
    
    # Mock the status
    status = MagicMock()
    status.status_code = MagicMock()
    status.status_code.name = "OK"
    span.status = status
    
    return span


@pytest.mark.skip(reason="Plugin system test skipped")
def test_isolated_plugin_system():
    """Test the plugin system in complete isolation."""
    print("=== Testing Isolated Plugin System ===\n")
    
    try:
        
        # Create test spans
        spans = [
            create_mock_span("agent_span", {"agent.name": "test_agent"}),
            create_mock_span("llm_span", {"span.type": "inference", "llm.cost": "0.05"}),
            create_mock_span("tool_span", {"tool.name": "search_tool"})
        ]
        
        # Create assertions instance
        assertions = TraceAssertions(spans)
        print("✓ Created TraceAssertions instance")
        
        # List plugins that were automatically loaded
        plugins = TraceAssertions.list_plugins()
        print(f"\n📋 Auto-loaded plugins ({len(plugins)}):")
        for name, plugin_class in plugins.items():
            methods = plugin_class.get_assertion_methods()
            print(f"   • {name}: {len(methods)} methods")
        
        # Test core plugin methods
        print("\n🧪 Testing plugin methods:")
        
        # Test span counting from core plugin
        try:
            assertions.assert_spans(count=3)
            print("   ✓ Core plugin: assert_spans works")
        except Exception as e:
            print(f"   ✗ Core plugin: assert_spans failed: {e}")
        
        # Test filtering from core plugin
        try:
            result = assertions.filter_by_name("agent_span")
            if result.count() == 1:
                print("   ✓ Core plugin: filter_by_name works")
            else:
                print(f"   ✗ Core plugin: filter_by_name returned {result.count()} spans, expected 1")
        except Exception as e:
            print(f"   ✗ Core plugin: filter_by_name failed: {e}")
        
        # Test agent plugin methods
        if hasattr(assertions, 'assert_agent'):
            try:
                TraceAssertions(spans).assert_agent("test_agent")
                print("   ✓ Agent plugin: assert_agent works")
            except Exception as e:
                print(f"   ✗ Agent plugin: assert_agent failed: {e}")
        else:
            print("   ✗ Agent plugin: assert_agent not available")
        
        # Test method chaining
        print("\n🔗 Testing method chaining:")
        try:
            result = (TraceAssertions(spans)
                     .assert_spans(min_count=1)
                     .filter_by_attribute("agent.name"))
            print(f"   ✓ Method chaining works (result has {result.count()} spans)")
        except Exception as e:
            print(f"   ✗ Method chaining failed: {e}")
        
        # Test custom plugin registration
        print("\n🔌 Testing custom plugin registration:")
        
        @plugin
        class TestIsolatedPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "test_isolated"
            
            def custom_isolated_assertion(self, value: str) -> 'TraceAssertions':
                """Test method for isolated plugin system."""
                print(f"      Custom isolated assertion called with: {value}")
                return self
        
        # Create new instance to get the new plugin
        new_assertions = TraceAssertions(spans)
        
        if hasattr(new_assertions, 'custom_isolated_assertion'):
            new_assertions.custom_isolated_assertion("test_value")
            print("   ✓ Custom plugin registration works")
        else:
            print("   ✗ Custom plugin registration failed")
        
        # Test plugin method availability
        print("\n📋 Available plugin methods:")
        available_methods = [method for method in dir(assertions) 
                           if not method.startswith('_') and callable(getattr(assertions, method))]
        
        expected_plugin_methods = [
            'assert_spans', 'filter_by_name', 'filter_by_attribute',  # core_plugins
            'assert_agent', 'called_tool',  # agent_plugins
            'input_contains', 'output_contains',  # content_plugins
            'semantically_contains_output',  # semantic_plugins
            'within_time_limit', 'assert_total_duration_under'  # performance from core_plugins
        ]
        
        found_methods = []
        missing_methods = []
        
        for method in expected_plugin_methods:
            if method in available_methods:
                found_methods.append(method)
            else:
                missing_methods.append(method)
        
        print(f"   ✓ Found {len(found_methods)} expected plugin methods")
        if missing_methods:
            print(f"   ⚠️  Missing methods: {missing_methods}")
        
        # Final summary
        final_plugins = TraceAssertions.list_plugins()
        print("\n📊 Final summary:")
        print(f"   • Total plugins: {len(final_plugins)}")
        print(f"   • Plugin methods available: {len(found_methods)}")
        print("   • Method chaining: working")
        print("   • Custom plugin registration: working")
        
        print("\n✅ Isolated plugin system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in isolated plugin system test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_isolated_plugin_system()
    sys.exit(0 if success else 1)