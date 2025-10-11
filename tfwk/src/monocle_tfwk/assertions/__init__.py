"""
Assertions module for TraceAssertions and plugins.

This module provides the TraceAssertions class and plugin system for validating
agent execution traces with fluent assertion APIs.
"""

# Import core classes first (no circular dependencies)
# Import trace utilities
from . import trace_utils
from .plugin_registry import (
    ConflictPolicy,
    TraceAssertionsPlugin,
    TraceAssertionsPluginRegistry,
    plugin,
)

# Import plugins to register them (this must come after TraceAssertions)
from .plugins import (
    agent,  # noqa: F401
    content,  # noqa: F401  
    core,  # noqa: F401
    llm,  # noqa: F401
    semantic,  # noqa: F401
)

# Import TraceAssertions (depends on plugin_registry but not on plugins)
from .trace_assertions import TraceAssertions

__all__ = [
    'TraceAssertions',
    'TraceAssertionsPlugin',
    'TraceAssertionsPluginRegistry', 
    'plugin',
    'ConflictPolicy',
    'trace_utils'
]
