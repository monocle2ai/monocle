"""A skipped execution can compute its response from the tool's arguments.

Test tools use this for mocks that answer each call differently. An entry in MONOCLE_SKIP_EXECUTIONS
may carry a ``response_fn``: it is called with the tool's arguments (parsed from the span's
``data.input`` event) and its return value is the response, with the usual placeholders filled in.
"""
import json
import unittest

from opentelemetry.context import attach, detach, set_value
from opentelemetry.sdk.trace import TracerProvider

from monocle_apptrace.instrumentation.common.constants import MONOCLE_SKIP_EXECUTIONS
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException, get_input_arguments_from_span

_tracer = TracerProvider().get_tracer("skip-execution-response-fn-test")


def _skip(entry, tool_input):
    """Run skip_execution on a tool span named search_flights whose input event holds tool_input."""
    token = attach(set_value(MONOCLE_SKIP_EXECUTIONS, {"search_flights": {
        "entity.type": "tool.adk", "span.type": "agentic.tool.invocation", **entry}}))
    try:
        with _tracer.start_as_current_span("search_flights") as span:
            span.set_attribute("span.type", "agentic.tool.invocation")
            span.set_attribute("entity.1.name", "search_flights")
            span.set_attribute("entity.1.type", "tool.adk")
            span.add_event("data.input", {"input": tool_input})
            return SpanHandler.skip_execution(span)
    finally:
        detach(token)


class TestSkipExecutionResponseFn(unittest.TestCase):

    def test_response_fn_gets_the_tool_arguments_and_its_result_is_returned(self):
        seen = []

        def respond(args):
            seen.append(args)
            return {"flights": [args["origin"] + "-" + args["destination"]]}

        skipped, response = _skip({"response_fn": respond}, json.dumps({"origin": "BLR", "destination": "BOM"}))
        self.assertTrue(skipped)
        self.assertEqual(response, {"flights": ["BLR-BOM"]})
        self.assertEqual(seen, [{"origin": "BLR", "destination": "BOM"}])

    def test_response_fn_result_gets_placeholders_filled(self):
        _, response = _skip({"response_fn": lambda args: "Found flights from {{origin}}"},
                            json.dumps({"origin": "SFO"}))
        self.assertEqual(response, "Found flights from SFO")

    def test_an_error_raised_by_response_fn_reaches_the_caller(self):
        def fail(args):
            raise MonocleSpanException("gateway timeout")

        with self.assertRaisesRegex(MonocleSpanException, "gateway timeout"):
            _skip({"response_fn": fail}, json.dumps({}))

    def test_response_fn_takes_precedence_over_a_fixed_response(self):
        _, response = _skip({"response_fn": lambda args: "computed", "response": "fixed"}, json.dumps({}))
        self.assertEqual(response, "computed")

    def test_a_fixed_response_is_unchanged_without_response_fn(self):
        skipped, response = _skip({"response": {"status": "ok"}}, json.dumps({"x": 1}))
        self.assertTrue(skipped)
        self.assertEqual(response, {"status": "ok"})


class TestInputArgumentsFromSpan(unittest.TestCase):

    def _events(self, tool_input):
        with _tracer.start_as_current_span("tool") as span:
            span.add_event("data.input", {"input": tool_input})
        return span.events

    def test_json_input_is_parsed(self):
        self.assertEqual(get_input_arguments_from_span(self._events('{"a": 1}')), {"a": 1})

    def test_python_repr_input_is_parsed(self):
        self.assertEqual(get_input_arguments_from_span(self._events("{'a': 1}")), {"a": 1})

    def test_input_that_is_not_a_mapping_is_passed_as_input(self):
        self.assertEqual(get_input_arguments_from_span(self._events("plain text")), {"input": "plain text"})

    def test_no_input_event_gives_an_empty_mapping(self):
        with _tracer.start_as_current_span("tool") as span:
            pass
        self.assertEqual(get_input_arguments_from_span(span.events), {})


if __name__ == "__main__":
    unittest.main()
