"""
Monocle Test Framework (monocle_tfwk) - A comprehensive testing framework for AI agents.

This package provides tools for testing AI agent workflows, validating traces,
and asserting on agent behavior with a fluent API.

Main Components:
- BaseAgentTest: Base class for agent testing
- TraceAssertions: Fluent API for trace validation  
- TraceQueryEngine: Query engine for trace analysis
- semantic_similarity: Semantic similarity utilities
"""

# Import main classes
from . import semantic_similarity
from .assertions import TraceAssertions
from .assertions.trace_utils import TraceQueryEngine
from .base_agent_test import BaseAgentTest

# Import validator for backward compatibility  
from .validator import MonocleValidator

__all__ = [
    'BaseAgentTest',
    'TraceAssertions', 
    'TraceQueryEngine',
    'semantic_similarity',
    'MonocleValidator'
]

__version__ = "0.1.0"
