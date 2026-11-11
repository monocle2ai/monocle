"""Evals and trace sources from a provider package are found by name."""
from unittest.mock import patch

from monocle_test_tools.evals import eval_manager
from monocle_test_tools.evals.base_eval import BaseEval
from monocle_test_tools.evals.eval_manager import get_eval_source_class, get_evaluator, get_supported_eval_sources
from monocle_test_tools.trace_sources import TraceSource, _trace_sources, get_trace_source


class _FakeEval(BaseEval):
    pass


class _FakeSource(TraceSource):
    name = "fake"


def test_eval_from_entry_point_is_found():
    targets = {"fake": f"{__name__}:_FakeEval"}
    with patch.object(eval_manager._eval_sources, "_entry_point_targets", return_value=targets):
        assert "fake" in get_supported_eval_sources()
        assert get_eval_source_class("fake") is _FakeEval
        assert isinstance(get_evaluator("fake", eval_options={}), _FakeEval)
    eval_manager._eval_sources._loaded.pop("fake", None)


def test_trace_source_from_entry_point_is_found():
    targets = {"fake": f"{__name__}:_FakeSource"}
    with patch.object(_trace_sources, "_entry_point_targets", return_value=targets):
        assert isinstance(get_trace_source("fake"), _FakeSource)
    _trace_sources._loaded.pop("fake", None)


def test_unknown_names_find_nothing():
    assert get_eval_source_class("no-such-eval") is None
    assert get_trace_source("no-such-source") is None
