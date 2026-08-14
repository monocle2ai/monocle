"""`exclude` opts a custom span out of capturing inputs and/or outputs."""

import json
import logging
import os
import shutil
import tempfile
import unittest

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from monocle_apptrace.instrumentation.common.constants import (
    CUSTOM_INSTRUMENTATION_FILE_NAME,
    CUSTOM_INSTRUMENTATION_FILE_PATH_ENV,
)
from monocle_apptrace.instrumentation.common.custom_span_processor import (
    build_custom_span_processor,
    normalize_exclusions,
)
from monocle_apptrace.instrumentation.common.instrumentor import (
    get_monocle_instrumentor,
    setup_monocle_telemetry,
)
from monocle_apptrace.instrumentation.common.method_wrappers import monocle_trace_method


def event_names(processor):
    return [event["name"] for event in processor["events"]]


def attribute_names(processor, event_name):
    for event in processor["events"]:
        if event["name"] == event_name:
            return [attribute["attribute"] for attribute in event["attributes"]]
    return []


class TestNormalizeExclusions(unittest.TestCase):
    def test_none_and_empty_exclude_nothing(self):
        for value in (None, "", [], ()):
            self.assertEqual(normalize_exclusions(value), frozenset())

    def test_accepts_a_bare_string(self):
        self.assertEqual(normalize_exclusions("inputs"), frozenset({"inputs"}))

    def test_accepts_an_iterable(self):
        self.assertEqual(
            normalize_exclusions(["inputs", "outputs"]), frozenset({"inputs", "outputs"})
        )

    def test_is_case_and_whitespace_insensitive(self):
        # Values from a YAML config must behave like values passed in code.
        self.assertEqual(normalize_exclusions([" Inputs ", "OUTPUTS"]),
                         frozenset({"inputs", "outputs"}))

    def test_unknown_name_warns_and_is_ignored(self):
        # A typo in a tracing option must not take down the traced application.
        with self.assertLogs(
            "monocle_apptrace.instrumentation.common.custom_span_processor",
            level=logging.WARNING,
        ) as captured:
            result = normalize_exclusions(["input", "outputs"])

        self.assertEqual(result, frozenset({"outputs"}))
        self.assertIn("input", "".join(captured.output))


class TestBuildCustomSpanProcessor(unittest.TestCase):
    def test_default_captures_inputs_and_outputs(self):
        processor = build_custom_span_processor()

        self.assertEqual(event_names(processor), ["data.input", "data.output"])
        self.assertIn("input", attribute_names(processor, "data.input"))
        self.assertIn("response", attribute_names(processor, "data.output"))

    def test_excluding_inputs_drops_the_input_event(self):
        processor = build_custom_span_processor("inputs")

        self.assertEqual(event_names(processor), ["data.output"])
        self.assertIn("response", attribute_names(processor, "data.output"))

    def test_excluding_outputs_drops_only_the_response_attribute(self):
        processor = build_custom_span_processor("outputs")

        self.assertEqual(event_names(processor), ["data.input", "data.output"])
        self.assertNotIn("response", attribute_names(processor, "data.output"))
        # error_code is not output data - it is whether the call failed, and an
        # excluded span must still be able to say something went wrong.
        self.assertIn("error_code", attribute_names(processor, "data.output"))

    def test_excluding_both_keeps_only_the_error_code(self):
        processor = build_custom_span_processor(["inputs", "outputs"])

        self.assertEqual(event_names(processor), ["data.output"])
        self.assertEqual(attribute_names(processor, "data.output"), ["error_code"])

    def test_the_module_default_is_unchanged(self):
        from monocle_apptrace.instrumentation.common.custom_span_processor import (
            CUSTOM_SPAN_PROCESSOR,
        )

        self.assertEqual(event_names(CUSTOM_SPAN_PROCESSOR), ["data.input", "data.output"])
        self.assertIn("input", attribute_names(CUSTOM_SPAN_PROCESSOR, "data.input"))
        self.assertIn("response", attribute_names(CUSTOM_SPAN_PROCESSOR, "data.output"))


class _ExporterCase(unittest.TestCase):
    """Shared telemetry setup: spans go to an in-memory exporter."""

    def setUp(self):
        existing = get_monocle_instrumentor()
        if existing is not None:
            try:
                existing.uninstrument()
            except Exception:
                pass

        self.memory_exporter = InMemorySpanExporter()

    def tearDown(self):
        try:
            if getattr(self, "instrumentor", None) is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            print("Uninstrument failed:", e)
        return super().tearDown()

    def captured_events(self, span_name):
        for span in self.memory_exporter.get_finished_spans():
            if span.name == span_name:
                return {event.name: dict(event.attributes) for event in span.events}
        available = [s.name for s in self.memory_exporter.get_finished_spans()]
        raise AssertionError(f"no span named {span_name!r}; got {available}")


class TestDecoratorExclude(_ExporterCase):
    def setUp(self):
        super().setUp()
        self.instrumentor = setup_monocle_telemetry(
            workflow_name="custom_span_exclude_test",
            span_processors=[SimpleSpanProcessor(self.memory_exporter)],
            union_with_default_methods=False,
        )

    def test_default_records_both(self):
        @monocle_trace_method(span_name="keeps_both")
        def handler(secret, flag=True):
            return {"value": secret}

        handler("s3cret")

        events = self.captured_events("keeps_both")
        self.assertIn("s3cret", events["data.input"]["input"])
        self.assertIn("s3cret", events["data.output"]["response"])

    def test_exclude_inputs_omits_the_arguments(self):
        @monocle_trace_method(span_name="no_inputs", exclude="inputs")
        def handler(secret):
            return "ok"

        handler("s3cret")

        events = self.captured_events("no_inputs")
        self.assertNotIn("data.input", events)
        self.assertIn("ok", events["data.output"]["response"])

    def test_exclude_outputs_omits_the_return_value(self):
        @monocle_trace_method(span_name="no_outputs", exclude="outputs")
        def handler(visible):
            return "s3cret-result"

        handler("visible-arg")

        events = self.captured_events("no_outputs")
        self.assertIn("visible-arg", events["data.input"]["input"])
        self.assertNotIn("response", events["data.output"])

    def test_exclude_both_still_records_the_span(self):
        @monocle_trace_method(span_name="no_io", exclude=["inputs", "outputs"])
        def handler(secret):
            return "s3cret-result"

        handler("s3cret-arg")

        events = self.captured_events("no_io")
        self.assertNotIn("data.input", events)
        self.assertNotIn("response", events["data.output"])
        # The span itself still exists - excluding data must not disable tracing.
        self.assertIn("data.output", events)

    def test_excluded_span_still_reports_an_error(self):
        @monocle_trace_method(span_name="failing", exclude=["inputs", "outputs"])
        def handler():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            handler()

        events = self.captured_events("failing")
        self.assertTrue(events["data.output"].get("error_code"))

    def test_exclude_does_not_change_the_return_value(self):
        @monocle_trace_method(span_name="returns", exclude=["inputs", "outputs"])
        def handler(value):
            return value * 2

        self.assertEqual(handler(21), 42)


class ConfigTarget:
    def do_work(self, secret: str) -> str:
        return f"result-for-{secret}"


class TestYamlExclude(_ExporterCase):
    def setUp(self):
        super().setUp()
        self.config_dir = tempfile.mkdtemp(prefix="monocle_exclude_")
        with open(os.path.join(self.config_dir, CUSTOM_INSTRUMENTATION_FILE_NAME), "w") as f:
            f.write(
                "instrument:\n"
                f"  - package: {ConfigTarget.__module__}\n"
                f"    class: {ConfigTarget.__name__}\n"
                "    method: do_work\n"
                "    sync: true\n"
                "    span_name: config_excluded\n"
                "    exclude:\n"
                "      - inputs\n"
                "      - outputs\n"
            )

        self._prev_env = os.environ.get(CUSTOM_INSTRUMENTATION_FILE_PATH_ENV)
        os.environ[CUSTOM_INSTRUMENTATION_FILE_PATH_ENV] = self.config_dir

        self.instrumentor = setup_monocle_telemetry(
            workflow_name="custom_span_exclude_yaml_test",
            span_processors=[SimpleSpanProcessor(self.memory_exporter)],
            union_with_default_methods=False,
        )

    def tearDown(self):
        if self._prev_env is None:
            os.environ.pop(CUSTOM_INSTRUMENTATION_FILE_PATH_ENV, None)
        else:
            os.environ[CUSTOM_INSTRUMENTATION_FILE_PATH_ENV] = self._prev_env
        shutil.rmtree(self.config_dir, ignore_errors=True)
        return super().tearDown()

    def test_exclude_from_the_config_file_is_honoured(self):
        self.assertEqual(ConfigTarget().do_work("s3cret"), "result-for-s3cret")

        events = self.captured_events("config_excluded")
        self.assertNotIn("data.input", events)
        self.assertNotIn("response", events["data.output"])
        self.assertNotIn("s3cret", json.dumps(events))


if __name__ == "__main__":
    unittest.main()
