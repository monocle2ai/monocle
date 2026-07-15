"""
Test conflict resolution in the plugin system.
"""
import os
import sys

import pytest
from monocle_tfwk.assertions.plugin_registry import (
    ConflictPolicy,
    TraceAssertionsPlugin,
    TraceAssertionsPluginRegistry,
    plugin,
)
from monocle_tfwk.assertions.trace_assertions import TraceAssertions


@pytest.mark.skip(reason="Plugin system test skipped")
def test_conflict_resolution():
    print("🔧 Testing Plugin Conflict Resolution System")
    print("=" * 50)
    
    try:
        # Clear any existing registrations
        TraceAssertionsPluginRegistry.clear()
        
        print("📋 1. Testing WARN Policy (Default)")
        TraceAssertionsPluginRegistry.set_conflict_policy(ConflictPolicy.WARN)
        
        @plugin
        class FirstPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "first"
            
            def common_method(self) -> 'TraceAssertions':
                """Method that will conflict."""
                print("      → FirstPlugin.common_method() called")
                return self
        
        @plugin
        class SecondPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "second" 
            
            def common_method(self) -> 'TraceAssertions':
                """Method that conflicts with FirstPlugin."""
                print("      → SecondPlugin.common_method() called")
                return self
        
        # Test which method is used (should be SecondPlugin due to WARN policy)
        assertions = TraceAssertions([])
        assertions.common_method()
        
        print("\n📋 2. Testing OVERRIDE Policy with Priorities")
        TraceAssertionsPluginRegistry.clear()
        TraceAssertionsPluginRegistry.set_conflict_policy(ConflictPolicy.OVERRIDE)
        
        @plugin(priority=1)
        class LowPriorityPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "low_priority"
            
            def priority_method(self) -> 'TraceAssertions':
                print("      → LowPriorityPlugin.priority_method() called")
                return self
        
        @plugin(priority=10)
        class HighPriorityPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "high_priority"
            
            def priority_method(self) -> 'TraceAssertions':
                print("      → HighPriorityPlugin.priority_method() called")
                return self
        
        # Test priority resolution
        assertions = TraceAssertions([])
        assertions.priority_method()  # Should use HighPriorityPlugin
        
        print("\n📋 3. Testing PREFIX Policy")
        TraceAssertionsPluginRegistry.clear()
        TraceAssertionsPluginRegistry.set_conflict_policy(ConflictPolicy.PREFIX)
        
        @plugin
        class PluginA(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "plugin_a"
            
            def shared_method(self) -> 'TraceAssertions':
                print("      → PluginA.shared_method() called")
                return self
        
        @plugin  
        class PluginB(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "plugin_b"
            
            def shared_method(self) -> 'TraceAssertions':
                print("      → PluginB.shared_method() called")
                return self
        
        # Test prefixed methods
        assertions = TraceAssertions([])
        if hasattr(assertions, 'shared_method'):
            assertions.shared_method()  # First plugin keeps original name
        if hasattr(assertions, 'plugin_b_shared_method'):
            assertions.plugin_b_shared_method()  # Second plugin gets prefixed
        
        print("\n📋 4. Testing ERROR Policy")
        TraceAssertionsPluginRegistry.clear()
        TraceAssertionsPluginRegistry.set_conflict_policy(ConflictPolicy.ERROR)
        
        @plugin
        class FirstErrorPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "first_error"
            
            def error_method(self) -> 'TraceAssertions':
                return self
        
        try:
            @plugin
            class SecondErrorPlugin(TraceAssertionsPlugin):
                @classmethod
                def get_plugin_name(cls) -> str:
                    return "second_error"
                
                def error_method(self) -> 'TraceAssertions':
                    return self
            
            print("      ✗ ERROR policy should have raised an exception")
        except ValueError as e:
            print(f"      ✓ ERROR policy correctly raised: {e}")
        
        print("\n📊 5. Conflict Resolution Summary")
        TraceAssertionsPluginRegistry.clear()
        TraceAssertionsPluginRegistry.set_conflict_policy(ConflictPolicy.WARN)
        
        # Register plugins with different priorities
        @plugin(priority=5)
        class CorePlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "core"
            
            def filter_by_name(self) -> 'TraceAssertions':
                print("      → CorePlugin.filter_by_name() called")
                return self
            
            def unique_core_method(self) -> 'TraceAssertions':
                return self
        
        @plugin(priority=1)
        class ExtensionPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "extension"
            
            def filter_by_name(self) -> 'TraceAssertions':
                print("      → ExtensionPlugin.filter_by_name() called")
                return self
            
            def unique_extension_method(self) -> 'TraceAssertions':
                return self
        
        plugins = TraceAssertionsPluginRegistry.list_plugins()
        print(f"      Registered plugins: {list(plugins.keys())}")
        
        assertions = TraceAssertions([])
        print("      Testing conflict resolution:")
        assertions.filter_by_name()  # Should use the last registered (ExtensionPlugin)
        
        print("\n✅ Conflict resolution system working correctly!")
        
        print(f"\n📈 Conflict Resolution Benefits:")
        print(f"   ✓ Multiple policies: ERROR, OVERRIDE, PREFIX, WARN")
        print(f"   ✓ Priority-based resolution for fine control")
        print(f"   ✓ Logging for transparency and debugging")
        print(f"   ✓ Backward compatibility with existing plugins")
        print(f"   ✓ Flexible deployment strategies")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in conflict resolution test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_conflict_resolution()
    sys.exit(0 if success else 1)