"""import_traces(load_spans=False) must be a pure fetch.

The A/B replay flow reads a recorded trace only to recover the input it was run
with, then runs a live agent that produces a *different* trace. If the fetch left
the source trace's fact id on the validator, post_test_cleanup would file the new
run's result against the old trace.
"""
import os
import shutil

import pytest

from monocle_test_tools import MonocleValidator

TRACE_ID = "e41d9435ad8b01f220bdca188d0867ec"
TRACE1_SPAN_COUNT = 14
TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


@pytest.fixture(name="trace_dir")
def trace_dir_fixture(tmp_path):
    """A directory holding trace1.json under the monocle file-naming convention."""
    dest = os.path.join(str(tmp_path), f"monocle_trace_svc_{TRACE_ID}_20250101.json")
    shutil.copyfile(os.path.join(TRACES_DIR, "trace1.json"), dest)
    return str(tmp_path)


@pytest.fixture(name="validator")
def validator_fixture():
    validator = MonocleValidator()
    validator.cleanup()
    yield validator
    validator.cleanup()


def test_returns_spans_when_loading(validator, trace_dir):
    spans = validator.import_traces(trace_source="file", id=TRACE_ID,
                                    trace_path=trace_dir)

    assert len(spans) == TRACE1_SPAN_COUNT


def test_loading_still_populates_validator(validator, trace_dir):
    validator.import_traces(trace_source="file", id=TRACE_ID, trace_path=trace_dir)

    assert len(validator.spans) == TRACE1_SPAN_COUNT
    assert validator._trace_source == "file"


def test_pure_fetch_returns_spans(validator, trace_dir):
    spans = validator.import_traces(trace_source="file", id=TRACE_ID,
                                    trace_path=trace_dir, load_spans=False)

    assert len(spans) == TRACE1_SPAN_COUNT


def test_pure_fetch_leaves_validator_spans_empty(validator, trace_dir):
    validator.import_traces(trace_source="file", id=TRACE_ID,
                            trace_path=trace_dir, load_spans=False)

    assert len(validator.spans) == 0


def test_pure_fetch_leaves_trace_source_unset(validator, trace_dir):
    validator._trace_source = None

    validator.import_traces(trace_source="file", id=TRACE_ID,
                            trace_path=trace_dir, load_spans=False)

    assert validator._trace_source is None


def test_pure_fetch_leaves_okahu_fact_fields_unset(validator, monkeypatch):
    """The okahu branch captures fact ids for result recording -- not in fetch mode."""
    from monocle_test_tools import validator as validator_module

    monkeypatch.setattr(validator_module.OkahuSpanLoader, "get_spans",
                        staticmethod(lambda **kwargs: []))
    validator._trace_source_fact_id = None
    validator._trace_source_fact_name = None
    validator._trace_source_workflow_name = None

    validator.import_traces(trace_source="okahu", id="t1", workflow_name="wf",
                            load_spans=False)

    assert validator._trace_source_fact_id is None
    assert validator._trace_source_fact_name is None
    assert validator._trace_source_workflow_name is None


def test_okahu_loading_still_captures_fact_fields(validator, monkeypatch):
    from monocle_test_tools import validator as validator_module

    monkeypatch.setattr(validator_module.OkahuSpanLoader, "get_spans",
                        staticmethod(lambda **kwargs: []))

    validator.import_traces(trace_source="okahu", id="t1", workflow_name="wf")

    assert validator._trace_source_fact_id == "t1"
    assert validator._trace_source_fact_name == "traces"
    assert validator._trace_source_workflow_name == "wf"
