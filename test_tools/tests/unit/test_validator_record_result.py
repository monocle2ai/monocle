"""Unit tests for MonocleValidator wiring of test-result recording.

Covers that ``import_traces`` captures the okahu fact details and that
``post_test_cleanup`` routes them to the resolved trace source's
``record_test_result``.
"""
from unittest.mock import MagicMock, patch

import pytest

from monocle_test_tools import MonocleValidator


@pytest.fixture
def validator():
    v = MonocleValidator()
    v.cleanup()
    yield v
    v.cleanup()
    v._trace_source = ""


class TestImportTracesCaptures:
    @pytest.mark.parametrize(
        "kwargs, expected_fact_name",
        [
            (dict(id="trace-1"), "traces"),
            (dict(id="sess-1", fact_name="session"), "agent_sessions"),
            (dict(id="scope-1", fact_name="scope", scope_name="test_id"), "test_id"),
        ],
    )
    def test_captures_okahu_fact_details(self, validator, kwargs, expected_fact_name):
        with patch("monocle_test_tools.validator.OkahuSpanLoader") as loader:
            loader.AGENT_SESSIONS_SCOPE = "agent_sessions"
            loader.get_spans.return_value = []
            loader.load_by_scope.return_value = []
            validator.import_traces(trace_source="okahu", workflow_name="my_app", **kwargs)

        assert validator._trace_source_fact_id == kwargs["id"]
        assert validator._trace_source_fact_name == expected_fact_name
        assert validator._trace_source_workflow_name == "my_app"


class TestPostTestCleanupRecords:
    def test_records_for_okahu_source(self, validator):
        validator._trace_source = "okahu"
        validator._trace_source_fact_id = "trace-1"
        validator._trace_source_fact_name = "traces"
        validator._trace_source_workflow_name = "my_app"

        fake_source = MagicMock()
        with patch("monocle_test_tools.validator.get_trace_source", return_value=fake_source) as gts:
            validator.post_test_cleanup(
                token=None, test_name="test_x", test_failed=True, skip_export=True
            )

        gts.assert_called_once_with("okahu")
        fake_source.record_test_result.assert_called_once()
        _, kwargs = fake_source.record_test_result.call_args
        assert kwargs["fact_id"] == "trace-1"
        assert kwargs["fact_name"] == "traces"
        assert kwargs["workflow_name"] == "my_app"
        assert kwargs["test_name"] == "test_x"
        assert kwargs["test_failed"] is True
        assert kwargs["exporters"] is validator.exporters

    def test_noop_for_non_okahu_source(self, validator):
        validator._trace_source = "file"
        # Real get_trace_source("file") -> None, so nothing should be recorded
        # and no error should be raised.
        validator.post_test_cleanup(
            token=None, test_name="test_x", test_failed=False, skip_export=True
        )

    def test_recording_failure_never_raises(self, validator):
        validator._trace_source = "okahu"
        validator._trace_source_fact_id = "trace-1"
        fake_source = MagicMock()
        fake_source.record_test_result.side_effect = RuntimeError("boom")
        with patch("monocle_test_tools.validator.get_trace_source", return_value=fake_source):
            # Must not raise.
            validator.post_test_cleanup(
                token=None, test_name="test_x", test_failed=False, skip_export=True
            )
