"""with_trace_source driven by a FluentTestCase.

Unlike the run_agent path this one *does* load the spans -- loading them is the
point of the call. `source` and `workflow_name` stay explicit configuration and
may accompany a test case; the identifying arguments may not, because the FactID
already supplies them.
"""
import pytest

from monocle_test_tools.fluent_api import TraceAssertion


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    TraceAssertion._assertion_errors = []
    TraceAssertion._okahu_filter = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._okahu_filter = None


@pytest.fixture(name="imported")
def imported_fixture(monkeypatch):
    """Record the kwargs with_trace_source forwards to import_traces."""
    seen = {}

    def fake_import_traces(self, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("monocle_test_tools.validator.MonocleValidator.import_traces",
                        fake_import_traces)
    return seen


@pytest.fixture(name="asserter")
def asserter_fixture():
    return TraceAssertion()


TESTCASE = {"input": {"fact_id": "trace1232", "fact_name": "trace"},
            "expected": {"evals": {"hallucination": "minor_hallucination"}}}


def test_explicit_source_wins_over_the_factid(asserter, imported):
    asserter.with_trace_source(source="okahu", testcase=TESTCASE, workflow_name="wf")

    assert imported["trace_source"] == "okahu"


def test_source_falls_back_to_the_factid(asserter, imported):
    asserter.with_trace_source(
        testcase={"input": {"fact_id": "t1", "fact_name": "trace", "source": "okahu"}},
        workflow_name="wf")

    assert imported["trace_source"] == "okahu"


def test_identifying_args_come_from_the_factid(asserter, imported):
    asserter.with_trace_source(source="okahu", testcase=TESTCASE, workflow_name="wf")

    assert imported["id"] == "trace1232"
    assert imported["fact_name"] == "trace"


def test_workflow_name_is_forwarded(asserter, imported):
    asserter.with_trace_source(source="okahu", testcase=TESTCASE, workflow_name="wf")

    assert imported["workflow_name"] == "wf"


def test_spans_are_loaded(asserter, imported):
    asserter.with_trace_source(source="okahu", testcase=TESTCASE, workflow_name="wf")

    assert imported.get("load_spans", True) is True


def test_custom_scope_fact_name_is_mapped(asserter, imported):
    asserter.with_trace_source(
        testcase={"input": {"fact_id": "t_123", "fact_name": "test_runid",
                            "source": "okahu"}},
        workflow_name="wf")

    assert imported["fact_name"] == "scope"
    assert imported["scope_name"] == "test_runid"


@pytest.mark.parametrize("conflicting", [
    {"id": "other"}, {"fact_name": "session"}, {"scope_name": "test_id"},
])
def test_identifying_args_alongside_testcase_raise(asserter, imported, conflicting):
    with pytest.raises(ValueError, match="cannot be combined with 'testcase'"):
        asserter.with_trace_source(source="okahu", testcase=TESTCASE,
                                   workflow_name="wf", **conflicting)


def test_non_factid_input_raises(asserter, imported):
    with pytest.raises(ValueError, match="FactID input"):
        asserter.with_trace_source(source="file", testcase={"input": "Book a flight"})


def test_default_source_is_still_local(asserter, imported):
    """No source and no testcase -> the pre-existing local default, no import."""
    asserter.with_trace_source()

    assert imported == {}


def test_existing_explicit_local_still_works(asserter, imported):
    asserter.with_trace_source("local")

    assert imported == {}
