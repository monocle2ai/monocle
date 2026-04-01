"""
Plugin system for TraceAssertions extensibility.

This module provides the base classes and registry for creating custom assertion plugins
that can extend the TraceAssertions class with domain-specific or application-specific
assertion methods.
"""
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Type

logger = logging.getLogger(__name__)


class ConflictPolicy:
    """Enumeration of conflict resolution policies."""
    ERROR = "error"          # Raise error on conflict (current behavior)
    OVERRIDE = "override"    # Allow newer plugin to override
    PREFIX = "prefix"        # Add plugin name prefix to conflicting methods
    WARN = "warn"           # Override with warning log


class TraceAssertionsPlugin(ABC):
    """
    Abstract base class for TraceAssertions plugins.
    
    Custom plugins should inherit from this class and implement assertion methods
    that will be dynamically added to TraceAssertions instances.
    
    Plugin methods should:
    1. Accept 'self' as first parameter (will be bound to TraceAssertions instance)
    2. Return 'TraceAssertions' for method chaining
    3. Follow the naming convention: assert_*, contains_*, with_*, etc.
    4. Include proper docstrings for documentation
    
    Example:
        class MyCustomPlugin(TraceAssertionsPlugin):
            def assert_custom_condition(self, param: str) -> 'TraceAssertions':
                '''Assert some custom condition.'''
                # Access spans via self._current_spans or self._all_spans
                # Perform assertions
                return self
    """
    
    @classmethod
    @abstractmethod
    def get_plugin_name(cls) -> str:
        """Return the unique name for this plugin."""
        pass
    
    @classmethod
    def get_assertion_methods(cls) -> Dict[str, Callable]:
        """
        Get all assertion methods from this plugin.
        
        Returns:
            Dict mapping method names to unbound method objects
        """
        methods = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip abstract methods and private methods
            if not name.startswith('_') and name not in ['get_plugin_name', 'get_assertion_methods']:
                methods[name] = method
        return methods


class TraceAssertionsPluginRegistry:
    """
    Registry for TraceAssertions plugins.
    
    Manages registration, discovery, and loading of plugins that extend
    TraceAssertions with custom assertion capabilities.
    """
    
    _plugins: Dict[str, Type[TraceAssertionsPlugin]] = {}
    _method_cache: Dict[str, Callable] = {}
    _method_owners: Dict[str, str] = {}  # method_name -> plugin_name mapping
    _plugin_priorities: Dict[str, int] = {}  # plugin_name -> priority
    _conflict_policy: str = ConflictPolicy.WARN  # Default policy
    
    @classmethod
    def set_conflict_policy(cls, policy: str) -> None:
        """Set the global conflict resolution policy."""
        if policy not in [ConflictPolicy.ERROR, ConflictPolicy.OVERRIDE, 
                         ConflictPolicy.PREFIX, ConflictPolicy.WARN]:
            raise ValueError(f"Invalid conflict policy: {policy}")
        cls._conflict_policy = policy

    @classmethod
    def register_plugin(cls, plugin_class: Type[TraceAssertionsPlugin], 
                       priority: int = 0) -> None:
        """
        Register a plugin class with conflict resolution.
        
        Args:
            plugin_class: Plugin class that inherits from TraceAssertionsPlugin
            priority: Plugin priority (higher numbers take precedence)
            
        Raises:
            ValueError: If plugin is invalid or conflict policy is ERROR and conflict exists
        """
        if not issubclass(plugin_class, TraceAssertionsPlugin):
            raise ValueError(f"Plugin {plugin_class} must inherit from TraceAssertionsPlugin")
        
        plugin_name = plugin_class.get_plugin_name()
        
        # Handle existing plugin replacement
        if plugin_name in cls._plugins:
            logger.info(f"TraceAssertions Plugin Registry: Replacing existing plugin '{plugin_name}'")
            cls._remove_plugin_methods(plugin_name)
        
        cls._plugins[plugin_name] = plugin_class
        cls._plugin_priorities[plugin_name] = priority
        
        # Register methods from this plugin with conflict resolution
        methods = plugin_class.get_assertion_methods()
        for method_name, method in methods.items():
            cls._register_method(method_name, method, plugin_name, priority)
    
    @classmethod
    def _register_method(cls, method_name: str, method: Callable, 
                        plugin_name: str, priority: int) -> None:
        """Register a method with conflict resolution."""
        if method_name in cls._method_cache:
            existing_plugin = cls._method_owners[method_name]
            existing_priority = cls._plugin_priorities[existing_plugin]
            
            if cls._conflict_policy == ConflictPolicy.ERROR:
                raise ValueError(
                    f"Method '{method_name}' conflicts with existing method from plugin '{existing_plugin}'"
                )
            elif cls._conflict_policy == ConflictPolicy.OVERRIDE:
                if priority >= existing_priority:
                    logger.info(f"TraceAssertions Plugin Registry: Method '{method_name}' overridden by higher priority plugin '{plugin_name}' (priority {priority}) replacing '{existing_plugin}' (priority {existing_priority})")
                    cls._method_cache[method_name] = method
                    cls._method_owners[method_name] = plugin_name
                else:
                    logger.info(f"TraceAssertions Plugin Registry: Method '{method_name}' kept from higher priority plugin '{existing_plugin}' (priority {existing_priority}), ignoring '{plugin_name}' (priority {priority})")
            elif cls._conflict_policy == ConflictPolicy.WARN:
                logger.info(f"TraceAssertions Plugin Registry: Method '{method_name}' conflict resolved - plugin '{plugin_name}' overrides '{existing_plugin}' (WARN policy)")
                cls._method_cache[method_name] = method
                cls._method_owners[method_name] = plugin_name
            elif cls._conflict_policy == ConflictPolicy.PREFIX:
                # Register with plugin prefix
                prefixed_name = f"{plugin_name}_{method_name}"
                cls._method_cache[prefixed_name] = method
                cls._method_owners[prefixed_name] = plugin_name
                logger.info(f"TraceAssertions Plugin Registry: Method '{method_name}' from plugin '{plugin_name}' registered as '{prefixed_name}' to avoid conflict with '{existing_plugin}'")
        else:
            cls._method_cache[method_name] = method
            cls._method_owners[method_name] = plugin_name
    
    @classmethod
    def _remove_plugin_methods(cls, plugin_name: str) -> None:
        """Remove all methods belonging to a specific plugin."""
        methods_to_remove = [
            method_name for method_name, owner in cls._method_owners.items()
            if owner == plugin_name
        ]
        for method_name in methods_to_remove:
            cls._method_cache.pop(method_name, None)
            cls._method_owners.pop(method_name, None)
    
    @classmethod
    def unregister_plugin(cls, plugin_name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
        """
        if plugin_name not in cls._plugins:
            return
        
        plugin_class = cls._plugins[plugin_name]
        methods = plugin_class.get_assertion_methods()
        
        # Remove methods from cache
        for method_name in methods:
            cls._method_cache.pop(method_name, None)
        
        # Remove plugin
        del cls._plugins[plugin_name]
    
    @classmethod
    def get_registered_plugins(cls) -> Dict[str, Type[TraceAssertionsPlugin]]:
        """Get all registered plugins."""
        return cls._plugins.copy()
    
    @classmethod
    def list_plugins(cls) -> Dict[str, Type[TraceAssertionsPlugin]]:
        """Alias for get_registered_plugins for convenience."""
        return cls.get_registered_plugins()
    
    @classmethod
    def get_available_methods(cls) -> Dict[str, Callable]:
        """Get all available plugin methods."""
        return cls._method_cache.copy()
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered plugins (mainly for testing)."""
        cls._plugins.clear()
        cls._method_cache.clear()
        cls._method_owners.clear()
        cls._plugin_priorities.clear()
    
    @classmethod
    def clear(cls) -> None:
        """Alias for clear_registry for convenience."""
        cls.clear_registry()


def plugin(plugin_class: Type[TraceAssertionsPlugin] = None, *, priority: int = 0):
    """
    Decorator to automatically register a plugin class with optional priority.
    
    Usage:
        @plugin  # Default priority 0
        class MyPlugin(TraceAssertionsPlugin):
            pass
            
        @plugin(priority=10)  # Higher priority
        class HighPriorityPlugin(TraceAssertionsPlugin):
            pass
    """
    def register_plugin_decorator(cls: Type[TraceAssertionsPlugin]) -> Type[TraceAssertionsPlugin]:
        TraceAssertionsPluginRegistry.register_plugin(cls, priority=priority)
        return cls
    
    # Support both @plugin and @plugin(priority=N) syntax
    if plugin_class is None:
        return register_plugin_decorator
    else:
        return register_plugin_decorator(plugin_class)