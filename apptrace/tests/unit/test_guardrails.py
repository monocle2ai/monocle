"""Guardrails: a check on a model call's input and output that stops the call when it triggers."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from openai.types.chat import ChatCompletion
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.trace.status import StatusCode

from monocle_apptrace.guardrails import GuardrailTriggered, guardrails_in_effect, load_guardrail
from monocle_apptrace.guardrails.stopping import stopped_traces
from monocle_apptrace.instructions import load_instruction
from monocle_apptrace.instructions.sending import add_system_message
from monocle_apptrace.instrumentation.common.constants import (
    GUARDRAIL_WATCHED, MONOCLE_GUARDRAILS_ENV, MONOCLE_GUARDRAILS_WATCH_ONLY_ENV, MONOCLE_INSTRUCTION_ENV,
    SPAN_TYPES, GUARDRAIL_NAME, GUARDRAIL_REASON, GUARDRAIL_SIDE,
)
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import get_error_message
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_iter_wrapper, task_wrapper

DATA = Path(__file__).parents[1] / "data"
STOP_ON_123456_FILE = str(DATA / "stop-on-123456.yaml")
STOP_ON_123456 = {MONOCLE_GUARDRAILS_ENV: STOP_ON_123456_FILE}
STOP_ANSWERS_WITH_123456 = {MONOCLE_GUARDRAILS_ENV: str(DATA / "stop-answers-with-123456.yaml")}
START_WITH_123456 = {MONOCLE_INSTRUCTION_ENV: str(DATA / "start-with-123456.yaml")}
QUESTION = [{"role": "user", "content": "Where is my booking?"}]
QUESTION_WITH_123456 = [{"role": "user", "content": "My code is 123456"}]
ANSWER_WITH_123456 = "123456 Your booking is confirmed."
STOPPED_ON_OUTPUT = "Kill switch triggered by guardrail 'stop-on-123456' on the output of a model call."

CHAT_INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "add_instruction": add_system_message,
    "is_auto_close": lambda kwargs: not kwargs.get("stream", False),
    "events": [
        {"name": "data.input", "attributes": [{
            "attribute": "input",
            "accessor": lambda arguments: [json.dumps({m["role"]: m["content"]}) for m in arguments["kwargs"]["messages"]],
        }]},
        {"name": "data.output", "attributes": [
            {"attribute": "error_code", "accessor": lambda arguments: get_error_message(arguments)},
            {"attribute": "response", "accessor": lambda arguments: str(arguments["result"])},
        ]},
    ],
}
CHAT_METHOD = {"package": "test", "object": "Chat", "method": "complete", "output_processor": CHAT_INFERENCE}
POSITIONAL_CHAT_METHOD = {**CHAT_METHOD, "output_processor": {
    "type": SPAN_TYPES.INFERENCE, "add_instruction": add_system_message, "events": []}}
AGENT_METHOD = {"package": "test", "object": "Agent", "method": "run", "output_processor": {
    "type": SPAN_TYPES.AGENTIC_INVOCATION, "attributes": [], "events": []}}
EMBEDDING_METHOD = {"package": "test", "object": "Embeddings", "method": "create", "output_processor": {
    "type": "embedding", "events": CHAT_INFERENCE["events"]}}


class RawResponse:
    """What an SDK returns in place of the model's response when the app asks for the raw HTTP response."""

    def __init__(self, request: str, parsed: str):
        self.request = request
        self.parsed = parsed


def openai_response(messages: list) -> ChatCompletion:
    choices = [{"index": index, "finish_reason": "stop", "message": {"role": "assistant", **message}}
               for index, message in enumerate(messages)]
    return ChatCompletion.model_validate({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini", "choices": choices})


def write_guardrail_file(folder: str, file_name: str, content: str) -> str:
    path = Path(folder) / file_name
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestFindingGuardrails(unittest.TestCase):
    def setUp(self):
        load_guardrail.cache_clear()
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        self._env_patcher.stop()

    def load(self, file_name: str, content: str):
        return load_guardrail(write_guardrail_file(self.folder, file_name, content))

    def test_no_guardrail_is_in_effect_by_default(self):
        self.assertEqual(guardrails_in_effect(), [])

    def test_guardrails_are_read_from_the_files_the_environment_lists(self):
        second = write_guardrail_file(self.folder, "no-keys.yaml", "name: no-keys\ndescription: d\npatterns: ['sk-[a-z]+']\n")
        os.environ[MONOCLE_GUARDRAILS_ENV] = f"{STOP_ON_123456_FILE}, {second}"
        self.assertEqual([guardrail.name for guardrail in guardrails_in_effect()], ["stop-on-123456", "no-keys"])

    def test_guardrail_checks_both_sides_unless_the_file_lists_them(self):
        both = self.load("both.yaml", "name: both\ndescription: d\npatterns: ['x']\n")
        output_only = self.load("out.yaml", "name: out\ndescription: d\nsides: [output]\npatterns: ['x']\n")
        self.assertEqual(both.sides, ("input", "output"))
        self.assertEqual(output_only.sides, ("output",))

    def test_reason_names_the_pattern_that_matched_by_its_place_in_the_file(self):
        guardrail = self.load("two.yaml", "name: two\ndescription: d\npatterns: ['aaa', 'b+']\n")
        self.assertEqual(guardrail.trigger_on("input", "abba").reason, "pattern 2")
        self.assertIsNone(guardrail.trigger_on("input", "cc"))

    def test_missing_file_raises_and_names_the_path(self):
        with self.assertRaisesRegex(ValueError, "nowhere.yaml was not found"):
            load_guardrail("nowhere.yaml")

    def test_file_without_patterns_raises(self):
        with self.assertRaisesRegex(ValueError, "has no 'patterns'"):
            self.load("empty.yaml", "name: empty\ndescription: d\n")

    def test_file_named_differently_from_its_name_raises(self):
        with self.assertRaisesRegex(ValueError, "must match the file name"):
            self.load("one.yaml", "name: another\ndescription: d\npatterns: ['x']\n")

    def test_pattern_that_is_not_a_regular_expression_raises_and_names_it(self):
        with self.assertRaisesRegex(ValueError, "pattern 2 is not a valid regular expression"):
            self.load("bad.yaml", "name: bad\ndescription: d\npatterns: ['ok', '(']\n")

    def test_side_that_is_not_input_or_output_raises(self):
        with self.assertRaisesRegex(ValueError, "checks 'tools'"):
            self.load("side.yaml", "name: side\ndescription: d\nsides: [tools]\npatterns: ['x']\n")

    def test_key_the_file_format_does_not_have_raises_and_names_it(self):
        with self.assertRaisesRegex(ValueError, "has a key 'side'"):
            self.load("typo.yaml", "name: typo\ndescription: d\nside: output\npatterns: ['x']\n")

    def test_name_that_is_not_text_raises(self):
        with self.assertRaisesRegex(ValueError, "has no 'name'"):
            self.load("123.yaml", "name: 123\ndescription: d\npatterns: ['x']\n")

    def test_file_that_is_not_yaml_raises_and_names_the_file(self):
        with self.assertRaisesRegex(ValueError, "broken.yaml is not valid YAML"):
            self.load("broken.yaml", "name: [broken\n")

    def test_pattern_that_would_stop_every_model_call_raises(self):
        with self.assertRaisesRegex(ValueError, "pattern 1 matches empty text"):
            self.load("all.yaml", "name: all\ndescription: d\npatterns: ['x*']\n")


class TestStoppingAModelCall(unittest.TestCase):
    def setUp(self):
        load_guardrail.cache_clear()
        load_instruction.cache_clear()
        stopped_traces.clear()
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()
        self.calls_sent = 0
        self.answer = "Your booking is confirmed."
        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def tearDown(self):
        self._env_patcher.stop()

    def model_method(self, **kwargs):
        self.calls_sent += 1
        return self.answer

    async def async_model_method(self, **kwargs):
        return self.model_method(**kwargs)

    def streaming_model_method(self, **kwargs):
        self.calls_sent += 1
        yield from self.answer

    def wrapped(self, method=CHAT_METHOD, wrapper=task_wrapper, tracer_provider=None):
        return wrapper((tracer_provider or self.tracer_provider).get_tracer("test"), SpanHandler(), method)

    def call_model(self, messages=QUESTION, method=CHAT_METHOD, tracer_provider=None, **kwargs):
        return self.wrapped(method, tracer_provider=tracer_provider)(self.model_method, None, (), {"messages": messages, **kwargs})

    def spans_of_type(self, span_type: str):
        return [span for span in self.exporter.get_finished_spans() if span.attributes.get("span.type") == span_type]

    def inference_span(self):
        return self.spans_of_type("inference")[-1]

    def output_of(self, span):
        return next(event for event in span.events if event.name == "data.output").attributes

    def test_call_is_sent_and_answered_when_no_guardrail_is_in_effect(self):
        self.assertEqual(self.call_model(QUESTION_WITH_123456), self.answer)
        self.assertNotIn(GUARDRAIL_NAME, self.inference_span().attributes)

    def test_call_that_triggers_nothing_is_sent_and_answered(self):
        os.environ.update(STOP_ON_123456)
        self.assertEqual(self.call_model(), self.answer)
        self.assertEqual(self.inference_span().status.status_code, StatusCode.OK)
        self.assertNotIn(GUARDRAIL_NAME, self.inference_span().attributes)

    def test_input_that_triggers_is_never_sent(self):
        os.environ.update(STOP_ON_123456)
        with self.assertRaises(GuardrailTriggered):
            self.call_model(QUESTION_WITH_123456)
        self.assertEqual(self.calls_sent, 0)

    def test_response_that_triggers_is_never_returned(self):
        os.environ.update(STOP_ON_123456)
        self.answer = ANSWER_WITH_123456
        with self.assertRaisesRegex(GuardrailTriggered, "on the output of a model call"):
            self.call_model()
        self.assertEqual(self.calls_sent, 1)

    def test_stopped_call_is_an_error_on_its_span_and_names_the_guardrail_and_the_reason(self):
        os.environ.update(STOP_ON_123456)
        self.answer = ANSWER_WITH_123456
        with self.assertRaises(GuardrailTriggered):
            self.call_model()
        span = self.inference_span()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.status.description, STOPPED_ON_OUTPUT)
        self.assertEqual(span.attributes[GUARDRAIL_NAME], "stop-on-123456")
        self.assertEqual(span.attributes[GUARDRAIL_SIDE], "output")
        self.assertEqual(span.attributes[GUARDRAIL_REASON], "pattern 1")
        self.assertEqual(self.output_of(span)["error_code"], "guardrail_triggered")
        self.assertNotIn(GUARDRAIL_WATCHED, span.attributes)

    def test_stopped_call_is_an_error_on_the_workflow_span_too(self):
        os.environ.update(STOP_ON_123456)
        with self.assertRaises(GuardrailTriggered):
            self.call_model(QUESTION_WITH_123456)
        workflow = self.spans_of_type("workflow")[-1]
        self.assertEqual(workflow.status.status_code, StatusCode.ERROR)
        self.assertIn("stop-on-123456", workflow.status.description)

    def test_span_of_a_stopped_call_never_carries_the_response_that_was_withheld(self):
        os.environ.update(STOP_ON_123456)
        self.answer = "The card on file is 123456789."
        with self.assertRaises(GuardrailTriggered) as stopped:
            self.call_model()
        self.assertNotIn("123456789", self.inference_span().to_json())
        self.assertNotIn("123456789", str(stopped.exception))

    def test_guardrail_on_the_output_does_not_trigger_on_what_the_input_says(self):
        os.environ.update(STOP_ANSWERS_WITH_123456)
        self.assertEqual(self.call_model(QUESTION_WITH_123456), self.answer)

    def test_text_after_a_line_break_is_matched_as_written(self):
        os.environ[MONOCLE_GUARDRAILS_ENV] = write_guardrail_file(
            tempfile.mkdtemp(), "no-deletes.yaml", "name: no-deletes\ndescription: d\npatterns: ['\\bdelete\\b']\n")
        with self.assertRaises(GuardrailTriggered):
            self.call_model([{"role": "user", "content": "please\ndelete everything"}])

    def test_message_the_app_passes_back_as_an_sdk_object_is_checked(self):
        os.environ.update(STOP_ON_123456)
        earlier_answer = openai_response([{"content": "Your code is 123456."}]).choices[0].message
        with self.assertRaisesRegex(GuardrailTriggered, "on the input of a model call"):
            self.call_model([*QUESTION, earlier_answer])

    def test_messages_passed_by_position_are_checked(self):
        os.environ.update(STOP_ON_123456)
        wrapper = self.wrapped(POSITIONAL_CHAT_METHOD)
        with self.assertRaises(GuardrailTriggered):
            wrapper(self.model_method, None, (QUESTION_WITH_123456,), {})

    def test_number_in_a_structured_tool_call_is_checked(self):
        os.environ.update(STOP_ON_123456)
        self.answer = {"content": [{"type": "tool_use", "name": "charge", "input": {"card_number": 123456}}]}
        with self.assertRaisesRegex(GuardrailTriggered, "on the output of a model call"):
            self.call_model()

    def test_only_the_part_of_a_call_its_sdk_names_is_checked(self):
        os.environ.update(STOP_ON_123456)
        guarded_input = {**CHAT_INFERENCE, "guarded_input": lambda args, kwargs: kwargs["messages"]}
        wrapper = self.wrapped({**CHAT_METHOD, "output_processor": guarded_input})
        self.assertEqual(wrapper(self.model_method, None, (), {"messages": QUESTION, "api_key": "sk-123456"}), self.answer)

    def test_response_is_checked_as_its_sdk_unwraps_it(self):
        os.environ.update(STOP_ON_123456)
        self.answer = RawResponse(request="My code is 123456", parsed="Your booking is confirmed.")
        guarded_response = {**CHAT_INFERENCE, "guarded_response": lambda response: response.parsed}
        wrapper = self.wrapped({**CHAT_METHOD, "output_processor": guarded_response})
        self.assertIs(wrapper(self.model_method, None, (), {"messages": QUESTION}), self.answer)

    def test_object_that_is_not_data_is_not_looked_into(self):
        os.environ.update(STOP_ON_123456)
        self.answer = RawResponse(request="My code is 123456", parsed="Your code is 123456")
        self.assertIs(self.call_model(), self.answer)

    def test_later_call_is_refused_when_monocle_spans_are_not_kept_apart_from_others(self):
        os.environ.update(STOP_ON_123456)
        with patch("monocle_apptrace.guardrails.stopping.get_current_monocle_span", return_value=INVALID_SPAN):
            with self.tracer_provider.get_tracer("test").start_as_current_span("turn"):
                with self.assertRaises(GuardrailTriggered):
                    self.call_model(QUESTION_WITH_123456)
                with self.assertRaises(GuardrailTriggered):
                    self.call_model()
        self.assertEqual(self.calls_sent, 0)

    def test_tool_call_the_model_asks_for_is_checked_beside_its_text(self):
        os.environ.update(STOP_ON_123456)
        self.answer = openai_response([
            {"content": "Cancelling it now.", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "cancel_booking", "arguments": '{"booking": "123456"}'}}]}])
        with self.assertRaisesRegex(GuardrailTriggered, "on the output of a model call"):
            self.call_model()

    def test_every_choice_in_a_response_is_checked(self):
        os.environ.update(STOP_ON_123456)
        self.answer = openai_response([{"content": "Your booking is confirmed."}, {"content": "Your code is 123456."}])
        with self.assertRaises(GuardrailTriggered):
            self.call_model()

    def test_every_later_model_call_in_a_stopped_trace_is_refused(self):
        os.environ.update(STOP_ON_123456)

        def agent_that_tries_again_after_an_error():
            try:
                self.call_model(QUESTION_WITH_123456)
            except GuardrailTriggered:
                pass
            return self.call_model()

        agent = self.wrapped(AGENT_METHOD)
        with self.assertRaisesRegex(GuardrailTriggered, "on the input of a model call"):
            agent(agent_that_tries_again_after_an_error, None, (), {})
        self.assertEqual(self.calls_sent, 0)
        first, refused = self.spans_of_type("inference")
        self.assertEqual(refused.attributes[GUARDRAIL_NAME], "stop-on-123456")
        self.assertEqual(refused.status.status_code, StatusCode.ERROR)

    def test_model_call_whose_inner_call_was_stopped_names_the_guardrail_too(self):
        os.environ.update(STOP_ON_123456)

        def library_that_wraps_the_error_of_the_call_beneath_it(**kwargs):
            try:
                return self.call_model(QUESTION_WITH_123456)
            except GuardrailTriggered as stopped:
                raise RuntimeError(f"LibraryError - {stopped}")

        outer = self.wrapped()
        with self.assertRaisesRegex(RuntimeError, "LibraryError - Kill switch triggered"):
            outer(library_that_wraps_the_error_of_the_call_beneath_it, None, (), {"messages": QUESTION})
        inner_span, outer_span = self.spans_of_type("inference")
        self.assertEqual(outer_span.attributes[GUARDRAIL_NAME], "stop-on-123456")
        self.assertEqual(outer_span.attributes[GUARDRAIL_SIDE], "input")
        self.assertEqual(outer_span.status.status_code, StatusCode.ERROR)

    def test_model_call_in_a_new_trace_is_sent(self):
        os.environ.update(STOP_ON_123456)
        with self.assertRaises(GuardrailTriggered):
            self.call_model(QUESTION_WITH_123456)
        self.assertEqual(self.call_model(), self.answer)

    def test_watch_only_marks_the_span_and_stops_nothing(self):
        os.environ.update(STOP_ON_123456)
        os.environ[MONOCLE_GUARDRAILS_WATCH_ONLY_ENV] = "true"
        self.assertEqual(self.call_model(QUESTION_WITH_123456), self.answer)
        self.assertEqual(self.call_model(), self.answer)
        watched, later = self.spans_of_type("inference")
        self.assertEqual(watched.status.status_code, StatusCode.OK)
        self.assertEqual(watched.attributes[GUARDRAIL_NAME], "stop-on-123456")
        self.assertEqual(watched.attributes[GUARDRAIL_SIDE], "input")
        self.assertTrue(watched.attributes[GUARDRAIL_WATCHED])
        self.assertNotIn(GUARDRAIL_NAME, later.attributes)

    def test_watch_only_marks_a_response_that_would_have_been_withheld(self):
        os.environ.update(STOP_ON_123456)
        os.environ[MONOCLE_GUARDRAILS_WATCH_ONLY_ENV] = "true"
        self.answer = ANSWER_WITH_123456
        self.assertEqual(self.call_model(), self.answer)
        self.assertEqual(self.inference_span().attributes[GUARDRAIL_SIDE], "output")
        self.assertTrue(self.inference_span().attributes[GUARDRAIL_WATCHED])

    def test_async_call_is_stopped_on_its_input(self):
        os.environ.update(STOP_ON_123456)
        wrapper = self.wrapped(wrapper=atask_wrapper)
        with self.assertRaisesRegex(GuardrailTriggered, "on the input of a model call"):
            asyncio.run(wrapper(self.async_model_method, None, (), {"messages": QUESTION_WITH_123456}))
        self.assertEqual(self.calls_sent, 0)

    def test_async_method_called_through_the_sync_wrapper_is_stopped_on_its_response(self):
        os.environ.update(STOP_ON_123456)
        self.answer = ANSWER_WITH_123456
        wrapper = self.wrapped()
        with self.assertRaises(GuardrailTriggered):
            asyncio.run(wrapper(self.async_model_method, None, (), {"messages": QUESTION}))

    def test_async_call_is_stopped_on_its_response(self):
        os.environ.update(STOP_ON_123456)
        self.answer = ANSWER_WITH_123456
        wrapper = self.wrapped(wrapper=atask_wrapper)
        with self.assertRaises(GuardrailTriggered):
            asyncio.run(wrapper(self.async_model_method, None, (), {"messages": QUESTION}))
        self.assertEqual(self.inference_span().attributes[GUARDRAIL_SIDE], "output")

    def test_unsampled_call_is_still_stopped(self):
        os.environ.update(STOP_ON_123456)
        with self.assertRaises(GuardrailTriggered):
            self.call_model(QUESTION_WITH_123456, tracer_provider=TracerProvider(sampler=ALWAYS_OFF))
        self.assertEqual(self.calls_sent, 0)

    def test_streamed_call_is_stopped_on_its_input(self):
        os.environ.update(STOP_ON_123456)
        wrapper = self.wrapped(wrapper=task_iter_wrapper)
        with self.assertRaises(GuardrailTriggered):
            list(wrapper(self.streaming_model_method, None, (), {"messages": QUESTION_WITH_123456}))
        self.assertEqual(self.calls_sent, 0)

    def test_streamed_response_is_not_checked(self):
        os.environ.update(STOP_ON_123456)
        self.answer = "123456"
        wrapper = self.wrapped(wrapper=task_iter_wrapper)
        self.assertEqual("".join(wrapper(self.streaming_model_method, None, (), {"messages": QUESTION})), "123456")

    def test_response_of_a_call_that_asked_for_a_stream_is_not_checked(self):
        os.environ.update(STOP_ON_123456)
        self.answer = "123456"
        self.assertEqual(self.call_model(stream=True), "123456")

    def test_call_that_is_not_a_model_call_is_not_checked(self):
        os.environ.update(STOP_ON_123456)
        self.assertEqual(self.call_model(QUESTION_WITH_123456, method=EMBEDDING_METHOD), self.answer)

    def test_guardrail_checks_the_call_with_the_instruction_added(self):
        os.environ.update(STOP_ON_123456)
        os.environ.update(START_WITH_123456)
        with self.assertRaisesRegex(GuardrailTriggered, "on the input of a model call"):
            self.call_model()
        span = self.inference_span()
        self.assertEqual(self.calls_sent, 0)
        self.assertEqual(span.attributes[GUARDRAIL_NAME], "stop-on-123456")

    def test_call_is_sent_unchecked_when_a_guardrail_file_is_missing(self):
        os.environ[MONOCLE_GUARDRAILS_ENV] = "nowhere.yaml"
        with self.assertLogs("monocle_apptrace.guardrails.stopping", level="WARNING"):
            self.assertEqual(self.call_model(QUESTION_WITH_123456), self.answer)


if __name__ == "__main__":
    unittest.main()
