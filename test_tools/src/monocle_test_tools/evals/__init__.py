"""
Evaluation modules for monocle test tools.

This package provides base, LLM-based, and non-LLM (deterministic)
evaluation implementations for validating AI agent responses and behavior.
"""

from monocle_apptrace.providers.moved import moved_attr
from .base_eval import BaseEval
from .bert_eval import BertScorerEval
from .regex_match_eval import RegexMatchEval
from .json_validity_eval import JSONValidityEval
from .keyword_presence_eval import KeywordPresenceEval
from .exact_match_eval import ExactMatchEval
from .eval_manager import get_evaluator

__all__ = [
    "BaseEval",
    "BertScorerEval",
    "RegexMatchEval",
    "JSONValidityEval",
    "KeywordPresenceEval",
    "ExactMatchEval",
    "get_evaluator",
]


def __getattr__(name):
    return moved_attr(__name__, name, {"OkahuEval": "monocle_test_tools.providers.okahu.okahu_eval"})
