"""
Comprehensive demonstration of the modular TraceAssertions plugin system.
"""
import sys
from unittest.mock import MagicMock

from monocle_tfwk.assertions import TraceAssertions
from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin


def create_mock_span(name="test_span", attributes=None, start_time=1000000000, end_time=1100000000):
    """Create a mock ReadableSpan for testing."""
    span = MagicMock()
    span.name = name
    span.attributes = attributes or {}
    span.start_time = start_time
    span.end_time = end_time
    
    # Mock the status
    status = MagicMock()
    status.status_code = MagicMock()
    status.status_code.name = "OK"
    span.status = status
    
    return span


def main():
    """Comprehensive demonstration of the plugin system."""
    print("🎯 TraceAssertions Modular Plugin System Demonstration")
    print("=" * 60)
    
    try:
        # Add assertions directory to path

        
        # Create comprehensive test data
        spans = [
            create_mock_span("workflow_start", {"workflow.name": "travel_planning", "input": "Plan a trip to Paris"}),
            create_mock_span("agent_task", {"agent.name": "travel_agent", "agent.type": "planner"}),
            create_mock_span("llm_call", {"span.type": "inference", "llm.model": "gpt-4", "llm.cost": "0.05", "llm.usage.input_tokens": "100", "llm.usage.output_tokens": "50"}),
            create_mock_span("flight_search", {"tool.name": "flight_search", "tool.result": "Found 5 flights"}),
            create_mock_span("agent_decision", {"agent.name": "travel_agent", "decision": "recommend_flight_1"}),
            create_mock_span("llm_summary", {"span.type": "inference", "llm.model": "gpt-4", "llm.cost": "0.03", "output": "I recommend Flight 1 to Paris"}),
            create_mock_span("workflow_end", {"workflow.status": "completed", "output": "Travel plan created successfully"})
        ]
        
        # Create assertions instance
        assertions = TraceAssertions(spans)
        
        print("\n📊 Plugin System Overview:")
        plugins = TraceAssertions.list_plugins()
        total_methods = 0
        for name, plugin_class in plugins.items():
            methods = plugin_class.get_assertion_methods()
            total_methods += len(methods)
            print(f"   • {name}: {len(methods)} methods")
        print(f"   → Total: {len(plugins)} plugins, {total_methods} methods")
        
        print("\n🔍 Core Plugin Demonstrations:")
        
        # Core span operations
        print("   1. Basic span counting and filtering:")
        try:
            assertions.assert_spans(count=7)
            agent_spans = assertions.filter_by_attribute("agent.name")
            print(f"      ✓ Total spans: 7, Agent spans: {agent_spans.count()}")
        except Exception as e:
            print(f"      ✗ Error: {e}")
        
        # Agent workflow validation
        print("   2. Agent workflow validation:")
        try:
            assertions.assert_agent("travel_agent")
            workflow_spans = assertions.filter_by_name("workflow_start").called_tool("flight_search")
            print(f"      ✓ Travel agent found, Workflow with tools: {workflow_spans.count()}")
        except Exception as e:
            print(f"      ✗ Error: {e}")
        
       
        # Content validation
        print("   3. Content pattern matching:")
        try:
            input_spans = assertions.input_contains("Paris")
            output_spans = assertions.output_contains("Flight")
            print(f"      ✓ Input with 'Paris': {input_spans.count()}, Output with 'Flight': {output_spans.count()}")
        except Exception as e:
            print(f"      ✗ Error: {e}")
        
        # Performance validation
        print("   4. Performance validation:")
        try:
            assertions.within_time_limit(1.0)  # 1 second
            print("      ✓ All spans within time limit")
        except Exception as e:
            print(f"      ✗ Error: {e}")
        
        print("\n🔗 Advanced Method Chaining:")
        try:
            result = (TraceAssertions(spans)
                     .assert_spans(min_count=5)
                     .filter_by_attribute("agent.name", "travel_agent")
                     .assert_spans(min_count=1)
                     .within_time_limit(2.0))
            
            print(f"      ✓ Complex chain: {result.count()} agent spans within time limit")
        except Exception as e:
            print(f"      ✗ Chaining error: {e}")
        
        print("\n🔌 Custom Plugin Creation:")
        
        @plugin
        class DemoCustomPlugin(TraceAssertionsPlugin):
            """Custom plugin for demonstration."""
            
            @classmethod
            def get_plugin_name(cls) -> str:
                return "demo_custom"
            
            def assert_demo_workflow_success(self) -> 'TraceAssertions':
                """Assert that a workflow completed successfully."""
                completed = False
                for span in self.spans:
                    if (span.name == "workflow_end" and 
                        span.attributes.get("workflow.status") == "completed"):
                        completed = True
                        break
                
                if not completed:
                    raise AssertionError("Workflow did not complete successfully")
                
                print("      → Custom assertion: Demo workflow completion verified")
                return self
            
            def count_by_type(self, span_type: str) -> int:
                """Count spans by type."""
                count = sum(1 for span in self.spans 
                          if span.attributes.get("span.type") == span_type)
                print(f"      → Custom method: Found {count} spans of type '{span_type}'")
                return count
        
        # Test custom plugin
        new_assertions = TraceAssertions(spans)
        new_assertions.assert_demo_workflow_success()
        new_assertions.count_by_type("inference")
        
        print("\n📈 Plugin System Benefits:")
        print("   ✓ Modular: Each plugin handles specific assertion categories")
        print("   ✓ Extensible: Easy to add new plugins with @plugin decorator")
        print("   ✓ Maintainable: Clean separation of concerns")
        print("   ✓ Compatible: Maintains existing TraceAssertions API")
        print("   ✓ Chainable: All methods support fluent interface")
        print("   ✓ Discoverable: Runtime plugin and method discovery")
        
        # Show available methods
        print(f"\n📋 All Available Methods ({total_methods}):")
        all_methods = []
        for name, plugin_class in plugins.items():
            methods = plugin_class.get_assertion_methods()
            for method_name in methods:
                all_methods.append(f"{name}.{method_name}")
        
        for i, method in enumerate(sorted(all_methods), 1):
            print(f"   {i:2d}. {method}")
        
        print("\n✅ Plugin system demonstration completed successfully!")
        print("\n🎉 TraceAssertions is now fully modular and extensible!")
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)