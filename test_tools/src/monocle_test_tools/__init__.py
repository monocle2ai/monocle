"""
Test tools for monocle AI agent validation.

This module provides classes and utilities for testing and validating AI agent behavior,
including agent invocations, tool usage, and response evaluation.
"""

from .validator import (
    MonocleValidator
)

from .schema import (
    TestCase,
    TestSpan,
    Entity,
    EntityType,
    SpanType,
    Evaluation,
    MockTool,
    ToolType,
)

from .evals import ( BaseEval, BertScorerEval, OkahuEval)
from .comparer import ( BaseComparer, BertScoreComparer, DefaultComparer)
from .runner import AgentRunner, get_agent_runner
from .fluent_api import TraceAssertion
from . import pytest_plugin
from . import gitutils

__all__ = [
    "MonocleValidator",
    "TestCase",
    "TestSpan",
    "Entity",
    "EntityType",
    "SpanType",
    "BaseEval",
    "Evaluation",
    "BertScorerEval",
    "OkahuEval",
    "BaseComparer",
    "BertScoreComparer",
    "DefaultComparer",
    "AgentRunner",
    "get_agent_runner",
    "MockTool",
    "ToolType",
    "TraceAssertion",
    "pytest_plugin",
    "gitutils"
]
