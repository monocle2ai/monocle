import datetime as _dt
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opentelemetry.sdk.resources import SERVICE_NAME

from monocle_apptrace.exporters import file_exporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter


class _Clock:
    """Controllable stand-in for `datetime`, returning real datetimes so subtraction/strftime work."""
    def __init__(self):
        self.t = _dt.datetime(2026, 1, 1, 0, 0, 0)

    def now(self):
        return self.t

    def advance(self, seconds):
        self.t += _dt.timedelta(seconds=seconds)


def _span(trace_id, span_id, has_parent=True, span_type=None):
    attrs = {"monocle_apptrace.version": "test"}
    if span_type:
        attrs["span.type"] = span_type
    return SimpleNamespace(
        context=SimpleNamespace(trace_id=trace_id, span_id=span_id),
        parent=SimpleNamespace() if has_parent else None,
        attributes=attrs,
        resource=SimpleNamespace(attributes={SERVICE_NAME: "idle_test"}),
    )


class TestFileExporterIdleTimeout(unittest.TestCase):
    """A trace handle should expire on idle time, not age since creation, so a long-running
    trace that keeps emitting spans stays in a single file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Isolate env vars to prevent config redirection away from test tmp directory.
        self.env_p = patch.dict(os.environ)
        self.env_p.start()
        os.environ.pop("MONOCLE_TRACE_OUTPUT_PATH", None)
        os.environ.pop("MONOCLE_FILE_PREFIX", None)
        self.clock = _Clock()
        self.p = patch.object(file_exporter, "datetime", self.clock)
        self.p.start()
        self.exp = FileSpanExporter(
            service_name="idle_test",
            out_path=self.tmp,
            formatter=lambda span: "{}",
        )

    def tearDown(self):
        self.exp.shutdown()
        self.p.stop()
        self.env_p.stop()

    def _files(self):
        return sorted(f for f in os.listdir(self.tmp) if f.endswith(".json"))

    def test_active_trace_stays_in_one_file(self):
        T = 0xABC
        # Spans keep arriving every 40s for an overall 120s trace (> 60s timeout).
        self.exp.export([_span(T, 1)])
        self.clock.advance(40)
        self.exp.export([_span(T, 2)])
        self.clock.advance(40)
        self.exp.export([_span(T, 3)])
        self.clock.advance(40)
        # Final batch carries the root: trace completes and the file closes.
        self.exp.export([_span(T, 4, has_parent=False, span_type="workflow"), _span(T, 5)])

        files = self._files()
        self.assertEqual(len(files), 1, f"expected one file for an active trace, got {files}")
        with open(os.path.join(self.tmp, files[0])) as f:
            self.assertTrue(f.read().rstrip().endswith("]"), "file should be closed")

    def test_idle_trace_is_still_expired(self):
        T, U = 0xAAA, 0xBBB
        self.exp.export([_span(T, 1)])
        self.assertIn(T, self.exp.file_handles)
        # 70s of silence, then a span for a different trace triggers cleanup.
        self.clock.advance(70)
        self.exp.export([_span(U, 9)])
        self.assertNotIn(T, self.exp.file_handles, "idle trace handle should have been closed")
        self.assertEqual(self.exp.last_trace_id, T)
        # T's file is closed (terminated with ']').
        with open(self.exp.last_file_processed) as f:
            self.assertTrue(f.read().rstrip().endswith("]"))


if __name__ == "__main__":
    unittest.main()
