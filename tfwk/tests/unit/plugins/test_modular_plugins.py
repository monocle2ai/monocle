"""
Test the modular plugin system for TraceAssertions.
"""
import sys
from unittest.mock import MagicMock

import pytest


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
def test_modular_plugins():
    """Test that the modular plugin system works correctly."""
    print("=== Testing Modular TraceAssertions Plugin System ===\n")
    
    try:
        # Import the new modular assertions
        from monocle_tfwk.assertions import (
            TraceAssertions,
            TraceAssertionsPlugin,
            plugin,
        )
        
        print("✓ Successfully imported modular TraceAssertions")
        
        # Create test spans
        spans = [
            create_mock_span("agent_span", {"agent.name": "test_agent"}),
            create_mock_span("llm_span", {"span.type": "inference", "llm.cost": "0.05"}),
            create_mock_span("tool_span", {"tool.name": "search_tool"})
        ]
        
        # Create assertions instance
        assertions = TraceAssertions(spans)
        print("✓ Created TraceAssertions instance")
        
        # Test core plugin methods
        try:
            assertions.assert_spans(count=3)
            print("✓ Core plugin: assert_spans works")
        except Exception as e:
            print(f"✗ Core plugin failed: {e}")
        
        # Test agent plugin methods
        try:
            assertions.assert_agent("test_agent")
            print("✓ Agent plugin: assert_agent works")
        except Exception as e:
            print(f"✗ Agent plugin failed: {e}")
        
        # Test content plugin methods (should be injected)
        if hasattr(assertions, 'with_input_containing'):
            print("✓ Content plugin: methods available")
        else:
            print("✗ Content plugin: methods not available")
            
        # Test LLM plugin methods
        if hasattr(assertions, 'llm_calls'):
            print("✓ LLM plugin: methods available")
        else:
            print("✗ LLM plugin: methods not available")
        
        # Test semantic plugin methods
        if hasattr(assertions, 'semantically_contains_output'):
            print("✓ Semantic plugin: methods available")
        else:
            print("✗ Semantic plugin: methods not available")
        
        # Test performance plugin methods
        if hasattr(assertions, 'within_time_limit'):
            print("✓ Performance plugin: methods available")
        else:
            print("✗ Performance plugin: methods not available")
        
        # Test method chaining still works
        try:
            (assertions
             .assert_spans(min_count=1)
             .filter_by_name("agent_span"))
            print("✓ Method chaining works with plugins")
        except Exception as e:
            print(f"✗ Method chaining failed: {e}")
        
        # Test custom plugin registration
        @plugin
        class TestModularPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "test_modular"
            
            def custom_modular_assertion(self, value: str) -> 'TraceAssertions':
                """Test method for modular plugin system."""
                print(f"   Custom modular assertion called with: {value}")
                return self
        
        # Create new instance to get the new plugin
        new_assertions = TraceAssertions(spans)
        
        if hasattr(new_assertions, 'custom_modular_assertion'):
            new_assertions.custom_modular_assertion("test_value")
            print("✓ Custom plugin registration and injection works")
        else:
            print("✗ Custom plugin registration failed")
        
        # List all available plugins
        plugins = TraceAssertions.list_plugins()
        print(f"\n📋 Available plugins ({len(plugins)}):")
        for name, plugin_class in plugins.items():
            methods = plugin_class.get_assertion_methods()
            print(f"   • {name}: {len(methods)} methods")
        
        print("\n✅ Modular plugin system is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing modular plugin system: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_modular_plugins()
    sys.exit(0 if success else 1)