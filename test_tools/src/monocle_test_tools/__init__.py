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
    MultiTurnTestCase,
    TestSpan,
    Entity,
    EntityType,
    SpanType,
    Evaluation,
    MockTool,
    ToolType,
)

from monocle_apptrace.providers.moved import moved_attr
from .evals import ( BaseEval, BertScorerEval)
from .comparer import ( BaseComparer, BertScoreComparer, DefaultComparer)
from . import trace_utils
from .runner import AgentRunner, get_agent_runner
from .fluent_api import TraceAssertion, setup_test_cases
from .csv_cases import CsvCase, load_cases_from_csv, monocle_csv_cases
from .scenarios import Scenario
from .test_generator import TestGenerator
from . import pytest_plugin
from . import gitutils

__all__ = [
    "MonocleValidator",
    "TestCase",
    "MultiTurnTestCase",
    "TestSpan",
    "Entity",
    "EntityType",
    "SpanType",
    "BaseEval",
    "Evaluation",
    "BertScorerEval",
    "BaseComparer",
    "BertScoreComparer",
    "DefaultComparer",
    "AgentRunner",
    "get_agent_runner",
    "MockTool",
    "ToolType",
    "TraceAssertion",
    "setup_test_cases",
    "CsvCase",
    "load_cases_from_csv",
    "monocle_csv_cases",
    "Scenario",
    "TestGenerator",
    "pytest_plugin",
    "gitutils"
]


def __getattr__(name):
    return moved_attr(__name__, name, {"OkahuEval": "monocle_test_tools.providers.okahu.okahu_eval"})