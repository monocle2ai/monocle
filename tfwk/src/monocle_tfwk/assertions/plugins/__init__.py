"""
Plugin modules for TraceAssertions.

This package contains all the assertion plugins that extend TraceAssertions
with domain-specific assertion capabilities. Importing this package registers
all plugins automatically.
"""

# Import all plugin modules to trigger registration
from . import agent, content, core, http, llm, semantic

__all__ = ['agent', 'content', 'core', 'http', 'llm', 'semantic']
