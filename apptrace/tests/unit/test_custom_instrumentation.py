import os
import shutil
import tempfile
import unittest

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from monocle_apptrace.instrumentation.common.constants import CUSTOM_INSTRUMENTATION_FILE_NAME, CUSTOM_INSTRUMENTATION_FILE_PATH_ENV
from monocle_apptrace.instrumentation.common.instrumentor import (
    get_monocle_instrumentor,
    setup_monocle_telemetry,
)


class CustomTarget:
    def do_work(self, value: int, dummy: str = "") -> int:
        return value * 2


class TestCustomInstrumentation(unittest.TestCase):

    def setUp(self):
        existing = get_monocle_instrumentor()
        if existing is not None:
            try:
                existing.uninstrument()
            except Exception:
                pass

        self.config_dir = tempfile.mkdtemp(prefix="monocle_custom_instr_")
        self.config_file = os.path.join(self.config_dir, CUSTOM_INSTRUMENTATION_FILE_NAME)
        with open(self.config_file, "w") as f:
            f.write(
                "instrument:\n"
                f"  - package: {CustomTarget.__module__}\n"
                f"    class: {CustomTarget.__name__}\n"
                "    method: do_work\n"
                "    sync: true\n"
            )

        self._prev_env = os.environ.get(CUSTOM_INSTRUMENTATION_FILE_PATH_ENV)
        os.environ[CUSTOM_INSTRUMENTATION_FILE_PATH_ENV] = self.config_dir

        self.memory_exporter = InMemorySpanExporter()
        self.instrumentor = setup_monocle_telemetry(
            workflow_name="custom_instr_test",
            span_processors=[SimpleSpanProcessor(self.memory_exporter)],
            union_with_default_methods=False,
        )

    def tearDown(self):
        try:
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            print("Uninstrument failed:", e)

        if self._prev_env is None:
            os.environ.pop(CUSTOM_INSTRUMENTATION_FILE_PATH_ENV, None)
        else:
            os.environ[CUSTOM_INSTRUMENTATION_FILE_PATH_ENV] = self._prev_env

        shutil.rmtree(self.config_dir, ignore_errors=True)
        return super().tearDown()

    def test_custom_yaml_method_emits_span(self):
        result = CustomTarget().do_work(21)
        self.assertEqual(result, 42)

        spans = self.memory_exporter.get_finished_spans()
        expected_name = f"{CustomTarget.__module__}.{CustomTarget.__name__}.do_work"
        matching = [s for s in spans if s.name == expected_name]
        self.assertTrue(
            matching,
            f"Expected a span named {expected_name!r}; got {[s.name for s in spans]}",
        )


if __name__ == "__main__":
    unittest.main()
