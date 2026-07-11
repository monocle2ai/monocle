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
from .span_loader import ( file_span_loader, okahu_span_loader, JSONSpanLoader, OkahuSpanLoader)
from . import trace_utils
from .runner import AgentRunner, get_agent_runner
from .fluent_api import TraceAssertion
from .test_generator import TestGenerator
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
    "TestGenerator",
    "pytest_plugin",
    "gitutils",
    "JSONSpanLoader",
    "OkahuSpanLoader",
    "file_span_loader",
    "okahu_span_loader",
]