"""Unit tests for the file-source span-loading changes in ``import_traces``.

Covers the ``trace_path`` argument added to ``MonocleValidator.import_traces``
(direct-file vs. directory search, id/path consistency check) and the
``trace_dir`` argument of ``JSONSpanLoader.find_trace_file``.
"""
import json
import os
import shutil

import pytest

from monocle_test_tools import MonocleValidator
from monocle_test_tools.file_span_loader import JSONSpanLoader

# trace_id of the spans stored in traces/trace1.json (without the 0x prefix).
TRACE_ID = "e41d9435ad8b01f220bdca188d0867ec"
TRACE1_SPAN_COUNT = 14

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


@pytest.fixture(autouse=True)
def _clean_validator_state():
    """MonocleValidator holds spans process-wide, so tests leak into each other.

    It matters here specifically: import_traces falls back to
    _get_current_trace_id() when no id is passed, so spans left behind by an
    earlier test in this file make the no-id case go looking for *that* trace and
    raise FileNotFoundError instead of the "'id' is required" ValueError. These
    tests pass alone and fail as a file without this.
    """
    MonocleValidator().cleanup()
    yield
    MonocleValidator().cleanup()


def _write_trace_file(directory, trace_id=TRACE_ID, service="svc", ts="20250101"):
    """Copy trace1.json into *directory* using the monocle file-naming convention.

    Returns the absolute path of the written file.
    """
    filename = f"monocle_trace_{service}_{trace_id}_{ts}.json"
    dest = os.path.join(str(directory), filename)
    shutil.copyfile(os.path.join(TRACES_DIR, "trace1.json"), dest)
    return dest


# --------------------------------------------------------------------------- #
# JSONSpanLoader.find_trace_file
# --------------------------------------------------------------------------- #

def test_find_trace_file_in_custom_dir(tmp_path):
    """find_trace_file locates a file by trace_id inside an explicit trace_dir."""
    expected = _write_trace_file(tmp_path)
    found = JSONSpanLoader.find_trace_file(TRACE_ID, trace_dir=str(tmp_path))
    assert found == expected


def test_find_trace_file_strips_0x_prefix(tmp_path):
    """A 0x-prefixed trace_id still matches the on-disk (prefix-less) filename."""
    expected = _write_trace_file(tmp_path)
    found = JSONSpanLoader.find_trace_file("0x" + TRACE_ID, trace_dir=str(tmp_path))
    assert found == expected


def test_find_trace_file_returns_none_when_missing(tmp_path):
    """No matching file → None (not an exception)."""
    _write_trace_file(tmp_path)
    assert JSONSpanLoader.find_trace_file("deadbeef", trace_dir=str(tmp_path)) is None


def test_find_trace_file_returns_most_recent(tmp_path):
    """When multiple files match, the most recently modified one is returned."""
    older = _write_trace_file(tmp_path, ts="20240101")
    newer = _write_trace_file(tmp_path, ts="20250101")
    # Force deterministic mtimes: older < newer.
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    assert JSONSpanLoader.find_trace_file(TRACE_ID, trace_dir=str(tmp_path)) == newer


# --------------------------------------------------------------------------- #
# import_traces — trace_path pointing directly at a file
# --------------------------------------------------------------------------- #

def test_import_traces_direct_file(tmp_path):
    """trace_path pointing at an existing file loads its spans without an id."""
    trace_file = _write_trace_file(tmp_path)
    validator = MonocleValidator()
    validator.import_traces(trace_source="file", trace_path=trace_file)
    assert len(validator.spans) == TRACE1_SPAN_COUNT


def test_import_traces_direct_file_matching_id(tmp_path):
    """A supplied id that appears in the file path is accepted."""
    trace_file = _write_trace_file(tmp_path)
    validator = MonocleValidator()
    validator.import_traces(trace_source="file", id=TRACE_ID, trace_path=trace_file)
    assert len(validator.spans) == TRACE1_SPAN_COUNT


def test_import_traces_direct_file_mismatched_id(tmp_path):
    """A supplied id absent from the file path raises ValueError."""
    trace_file = _write_trace_file(tmp_path)
    validator = MonocleValidator()
    with pytest.raises(ValueError, match="does not match the given trace_id"):
        validator.import_traces(
            trace_source="file", id="deadbeefdeadbeef", trace_path=trace_file
        )


# --------------------------------------------------------------------------- #
# import_traces — trace_path pointing at a directory
# --------------------------------------------------------------------------- #

def test_import_traces_directory_search(tmp_path):
    """trace_path as a directory triggers a find_trace_file search by id."""
    _write_trace_file(tmp_path)
    validator = MonocleValidator()
    validator.import_traces(
        trace_source="file", id=TRACE_ID, trace_path=str(tmp_path)
    )
    assert len(validator.spans) == TRACE1_SPAN_COUNT


def test_import_traces_directory_not_found(tmp_path):
    """No matching file in the directory raises FileNotFoundError naming the dir."""
    validator = MonocleValidator()
    with pytest.raises(FileNotFoundError) as exc:
        validator.import_traces(
            trace_source="file", id="deadbeef", trace_path=str(tmp_path)
        )
    assert str(tmp_path) in str(exc.value)


# --------------------------------------------------------------------------- #
# import_traces — argument validation
# --------------------------------------------------------------------------- #

def test_import_traces_requires_id_without_path():
    """File source with neither id, trace_path, nor current spans → ValueError."""
    validator = MonocleValidator()
    with pytest.raises(ValueError, match="'id' is required"):
        validator.import_traces(trace_source="file")


def test_import_traces_unsupported_source():
    """An unknown trace_source is rejected before any loading occurs."""
    validator = MonocleValidator()
    with pytest.raises(ValueError, match="Unsupported trace_source"):
        validator.import_traces(trace_source="s3", id=TRACE_ID)


def test_import_traces_file_source_rejects_non_trace_fact(tmp_path):
    """File source only supports fact_name='trace'."""
    trace_file = _write_trace_file(tmp_path)
    validator = MonocleValidator()
    with pytest.raises(ValueError, match="Only fact_name='trace' is supported"):
        validator.import_traces(
            trace_source="file", trace_path=trace_file, fact_name="session"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
