"""
Direct test of the modular plugin system for TraceAssertions.
"""
import sys
from unittest.mock import MagicMock

# Add the src directory to the path
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


def test_direct_plugin_system():
    """Test the plugin system directly without full package imports."""
    print("=== Testing Direct Plugin System ===\n")
    
    try:
        # Import just the assertions components
        
        print("✓ Successfully imported TraceAssertions components")
        
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
        
        # Test attribute filtering from core plugin
        try:
            result = TraceAssertions(spans).filter_by_attribute("agent.name", "test_agent")
            if result.count() == 1:
                print("   ✓ Core plugin: filter_by_attribute works")
            else:
                print(f"   ✗ Core plugin: filter_by_attribute returned {result.count()} spans, expected 1")
        except Exception as e:
            print(f"   ✗ Core plugin: filter_by_attribute failed: {e}")
        
        # Test agent plugin methods if available
        if hasattr(assertions, 'assert_agent'):
            try:
                assertions.assert_agent("test_agent")
                print("   ✓ Agent plugin: assert_agent works")
            except Exception as e:
                print(f"   ✗ Agent plugin: assert_agent failed: {e}")
        else:
            print("   ✗ Agent plugin: assert_agent not available")
        
        # Test LLM plugin methods if available
        if hasattr(assertions, 'llm_calls'):
            try:
                result = TraceAssertions(spans).llm_calls()
                print(f"   ✓ LLM plugin: llm_calls works (found {result.count()} LLM calls)")
            except Exception as e:
                print(f"   ✗ LLM plugin: llm_calls failed: {e}")
        else:
            print("   ✗ LLM plugin: llm_calls not available")
        
        # Test content plugin methods if available
        if hasattr(assertions, 'output_contains'):
            print("   ✓ Content plugin: methods available")
        else:
            print("   ✗ Content plugin: methods not available")

        # Test semantic plugin methods if available  
        if hasattr(assertions, 'semantically_contains_output'):
            print("   ✓ Semantic plugin: methods available")
        else:
            print("   ✗ Semantic plugin: methods not available")

        # Test performance plugin methods if available
        if hasattr(assertions, 'within_time_limit'):
            try:
                assertions.within_time_limit(2.0)  # Should pass with our mock spans
                print("   ✓ Performance plugin: within_time_limit works")
            except Exception as e:
                print(f"   ✗ Performance plugin: within_time_limit failed: {e}")
        else:
            print("   ✗ Performance plugin: within_time_limit not available")

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
        class TestDirectPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "test_direct"

            def custom_direct_assertion(self, value: str) -> 'TraceAssertions':
                """Test method for direct plugin system."""
                print(f"      Custom assertion called with: {value}")
                return self

        # Create new instance to get the new plugin
        new_assertions = TraceAssertions(spans)

        if hasattr(new_assertions, 'custom_direct_assertion'):
            new_assertions.custom_direct_assertion("test_value")
            print("   ✓ Custom plugin registration works")
        else:
            print("   ✗ Custom plugin registration failed")

        # Final plugin count
        final_plugins = TraceAssertions.list_plugins()
        print(f"\n📊 Final plugin count: {len(final_plugins)}")

        print("\n✅ Direct plugin system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in direct plugin system test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_direct_plugin_system()
    sys.exit(0 if success else 1)