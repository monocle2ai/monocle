"""
Monocle Test Framework (tfwk) - Pytest-style agent testing framework.

This package provides an intuitive way to test AI agents using pytest fixtures
and base test classes, inspired by AgentiTest but designed for trace-based validation.
"""

from . import trace_utils
from .agent_test import (
    BaseAgentTest,
    TraceAssertions,
    agent_test_context,
    assert_agent_sequence,
    assert_output_contains,
    assert_tool_invocations,
    trace_validator,
)
from .semantic_similarity import (
    SemanticSimilarityChecker,
    semantic_similarity,
    semantic_similarity_score,
)
from .trace_utils import TraceQueryEngine
from .validator import MonocleValidator

__version__ = "0.1.0"

__all__ = [
    "BaseAgentTest",
    "TraceAssertions", 
    "TraceQueryEngine",
    "MonocleValidator",
    "SemanticSimilarityChecker",
    "semantic_similarity",
    "semantic_similarity_score",
    "trace_validator",
    "agent_test_context",
    "assert_agent_sequence",
    "assert_tool_invocations",
    "assert_output_contains",
    "trace_utils"
]