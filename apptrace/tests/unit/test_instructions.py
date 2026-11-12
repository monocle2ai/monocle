"""An instruction is found by name, added to every model call, and shown in the trace."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opentelemetry.context import attach, detach, set_value
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from monocle_apptrace.instructions import instruction_in_effect, load_instruction
from monocle_apptrace.instructions.sending import add_system_message, instruction_before
from monocle_apptrace.instrumentation.common.constants import (
    MONOCLE_INSTRUCTION, MONOCLE_INSTRUCTION_ENV, MONOCLE_SUPPRESS_SPANS, SENT_INSTRUCTION, SPAN_TYPES,
)
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.anthropic import _helper as anthropic
from monocle_apptrace.instrumentation.metamodel.botocore import _helper as bedrock
from monocle_apptrace.instrumentation.metamodel.gemini import _helper as gemini
from monocle_apptrace.instrumentation.metamodel.litellm import _helper as litellm
from monocle_apptrace.instrumentation.metamodel.openai import _helper as openai

TEXT = "Only help with bookings."
START_WITH_123456_FILE = str(Path(__file__).parents[1] / "data" / "start-with-123456.yaml")
START_WITH_123456 = {MONOCLE_INSTRUCTION_ENV: START_WITH_123456_FILE}
DEVELOPER_MESSAGES = [{"role": "system", "content": "You book travel."}, {"role": "user", "content": "A poem, please"}]

CHAT_INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "add_instruction": add_system_message,
    "events": [{"name": "data.input", "attributes": [{
        "attribute": "input",
        "accessor": lambda arguments: [json.dumps({m["role"]: m["content"]}) for m in arguments["kwargs"]["messages"]],
    }]}],
}
CHAT_METHOD = {"package": "test", "object": "Chat", "method": "complete", "output_processor": CHAT_INFERENCE}


def write_instruction_file(folder: str, file_name: str, content: str) -> str:
    path = Path(folder) / file_name
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestFindingAnInstruction(unittest.TestCase):
    def setUp(self):
        load_instruction.cache_clear()
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()

    def tearDown(self):
        self._env_patcher.stop()

    def test_no_instruction_is_in_effect_by_default(self):
        self.assertIsNone(instruction_in_effect())

    def test_instruction_is_read_from_the_file_the_environment_names(self):
        os.environ.update(START_WITH_123456)
        instruction = instruction_in_effect()
        self.assertEqual(instruction.name, "start-with-123456")
        self.assertEqual(instruction.text, "Begin every response with the digits 123456, before anything else.")

    def test_missing_file_raises_and_names_the_path(self):
        with self.assertRaisesRegex(ValueError, "config/house-style.yaml was not found"):
            load_instruction("config/house-style.yaml")

    def test_file_without_instruction_text_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_instruction_file(folder, "empty.yaml", "name: empty\ndescription: Nothing here.\n")
            with self.assertRaisesRegex(ValueError, "has no 'instruction'"):
                load_instruction(path)

    def test_file_named_differently_from_its_name_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_instruction_file(folder, "one.yaml", "name: two\ndescription: d\ninstruction: i\n")
            with self.assertRaisesRegex(ValueError, "must match the file name"):
                load_instruction(path)

    def test_context_is_asked_before_the_environment(self):
        os.environ.update(START_WITH_123456)
        with tempfile.TemporaryDirectory() as folder:
            path = write_instruction_file(folder, "be-brief.yaml",
                                          "name: be-brief\ndescription: Short answers.\ninstruction: Answer in one sentence.\n")
            token = attach(set_value(MONOCLE_INSTRUCTION, path))
            try:
                self.assertEqual(instruction_in_effect().text, "Answer in one sentence.")
            finally:
                detach(token)


class TestPlacingAnInstruction(unittest.TestCase):
    def test_chat_call_gets_it_as_the_first_system_message(self):
        instructed = add_system_message({"messages": DEVELOPER_MESSAGES}, TEXT)
        self.assertEqual(instructed["messages"], [{"role": "system", "content": TEXT}, *DEVELOPER_MESSAGES])

    def test_developer_messages_are_left_as_they_were(self):
        messages = list(DEVELOPER_MESSAGES)
        add_system_message({"messages": messages}, TEXT)
        self.assertEqual(messages, DEVELOPER_MESSAGES)

    def test_it_is_not_added_twice(self):
        once = add_system_message({"messages": DEVELOPER_MESSAGES}, TEXT)
        self.assertEqual(add_system_message(once, TEXT)["messages"], once["messages"])

    def test_call_without_messages_is_not_instructed(self):
        self.assertIsNone(add_system_message({"model": "m"}, TEXT))

    def test_system_prompt_that_starts_with_the_same_words_is_still_instructed(self):
        self.assertEqual(instruction_before("Be brief.", "Be brief. And polite."), "Be brief.\n\nBe brief. And polite.")

    def test_system_prompt_that_is_not_text_is_never_replaced(self):
        self.assertIsNone(openai.add_instruction({"input": "A poem, please", "instructions": object()}, TEXT))

    def test_openai_responses_gets_it_ahead_of_instructions(self):
        instructed = openai.add_instruction({"input": "A poem, please", "instructions": "You book travel."}, TEXT)
        self.assertEqual(instructed["instructions"], f"{TEXT}\n\nYou book travel.")

    def test_openai_responses_without_instructions_gets_it_alone(self):
        self.assertEqual(openai.add_instruction({"input": "A poem, please"}, TEXT)["instructions"], TEXT)

    def test_openai_call_without_a_prompt_is_not_instructed(self):
        self.assertIsNone(openai.add_instruction({"model": "m"}, TEXT))

    def test_anthropic_gets_it_ahead_of_system_text(self):
        instructed = anthropic.add_instruction({"messages": DEVELOPER_MESSAGES[1:], "system": "You book travel."}, TEXT)
        self.assertEqual(instructed["system"], f"{TEXT}\n\nYou book travel.")

    def test_anthropic_gets_it_ahead_of_system_blocks(self):
        developer_block = {"type": "text", "text": "You book travel."}
        instructed = anthropic.add_instruction({"messages": DEVELOPER_MESSAGES[1:], "system": [developer_block]}, TEXT)
        self.assertEqual(instructed["system"], [{"type": "text", "text": TEXT}, developer_block])

    def test_anthropic_system_blocks_given_as_a_tuple_are_kept(self):
        developer_block = {"type": "text", "text": "You book travel."}
        instructed = anthropic.add_instruction({"messages": DEVELOPER_MESSAGES[1:], "system": (developer_block,)}, TEXT)
        self.assertEqual(instructed["system"], [{"type": "text", "text": TEXT}, developer_block])

    def test_gemini_gets_it_in_a_new_config(self):
        self.assertEqual(gemini.add_instruction({"contents": "A poem, please"}, TEXT)["config"], {"system_instruction": TEXT})

    def test_gemini_gets_it_ahead_of_system_instruction_in_a_dict_config(self):
        config = {"system_instruction": "You book travel.", "temperature": 0}
        instructed = gemini.add_instruction({"contents": "A poem, please", "config": config}, TEXT)
        self.assertEqual(instructed["config"], {"system_instruction": f"{TEXT}\n\nYou book travel.", "temperature": 0})
        self.assertEqual(config["system_instruction"], "You book travel.")

    def test_gemini_dict_config_keeps_the_key_spelling_the_developer_used(self):
        instructed = gemini.add_instruction({"contents": "A poem, please", "config": {"systemInstruction": "You book travel."}}, TEXT)
        self.assertEqual(instructed["config"], {"systemInstruction": f"{TEXT}\n\nYou book travel."})

    def test_gemini_config_object_is_copied_not_changed(self):
        from google.genai import types
        config = types.GenerateContentConfig(system_instruction="You book travel.")
        instructed = gemini.add_instruction({"contents": "A poem, please", "config": config}, TEXT)
        self.assertEqual(instructed["config"].system_instruction, f"{TEXT}\n\nYou book travel.")
        self.assertEqual(config.system_instruction, "You book travel.")

    def test_gemini_gets_it_ahead_of_the_parts_of_a_content(self):
        from google.genai import types
        developer_part = types.Part(text="You book travel.")
        config = types.GenerateContentConfig(system_instruction=types.Content(parts=[developer_part]))
        instructed = gemini.add_instruction({"contents": "A poem, please", "config": config}, TEXT)
        self.assertEqual(instructed["config"].system_instruction, [TEXT, developer_part])

    def test_gemini_call_that_uses_cached_content_is_not_instructed(self):
        self.assertIsNone(gemini.add_instruction({"contents": "A poem, please", "config": {"cached_content": "cachedContents/abc"}}, TEXT))

    def test_openai_agents_model_leaves_the_instruction_to_the_responses_call_beneath_it(self):
        from monocle_apptrace.instrumentation.metamodel.openai.methods import OPENAI_METHODS
        agents_methods = [method for method in OPENAI_METHODS if method["package"].startswith("agents.")]
        self.assertTrue(agents_methods)
        for method in agents_methods:
            self.assertNotIn("add_instruction", method["output_processor"])

    def test_bedrock_converse_gets_it_as_the_first_system_block(self):
        instructed = bedrock.add_instruction({"messages": [], "system": [{"text": "You book travel."}]}, TEXT)
        self.assertEqual(instructed["system"], [{"text": TEXT}, {"text": "You book travel."}])

    def test_bedrock_invoke_model_is_not_instructed(self):
        self.assertIsNone(bedrock.add_instruction({"modelId": "m", "body": "{}"}, TEXT))

    def test_litellm_azure_async_gets_it_under_data(self):
        instructed = litellm.add_instruction({"data": {"messages": DEVELOPER_MESSAGES}}, TEXT)
        self.assertEqual(instructed["data"]["messages"][0], {"role": "system", "content": TEXT})

    def test_litellm_responses_gets_it_in_request_params(self):
        kwargs = {"input": "A poem, please", "response_api_optional_request_params": {"instructions": "You book travel."}}
        instructed = litellm.add_responses_instruction(kwargs, TEXT)
        self.assertEqual(instructed["response_api_optional_request_params"]["instructions"], f"{TEXT}\n\nYou book travel.")


class TestSendingAModelCall(unittest.TestCase):
    def setUp(self):
        load_instruction.cache_clear()
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()
        self.sent_messages = None
        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def tearDown(self):
        self._env_patcher.stop()

    def model_method(self, *args, **kwargs):
        self.sent_messages = kwargs.get("messages") or args[0]
        return "ok"

    async def async_model_method(self, **kwargs):
        return self.model_method(**kwargs)

    def failing_model_method(self, **kwargs):
        raise RuntimeError("model is down")

    def call_model(self, tracer_provider=None):
        tracer = (tracer_provider or self.tracer_provider).get_tracer("test")
        wrapper = task_wrapper(tracer, SpanHandler(), CHAT_METHOD)
        return wrapper(self.model_method, None, (), {"messages": DEVELOPER_MESSAGES})

    def inference_span(self):
        return next(span for span in self.exporter.get_finished_spans() if span.attributes.get("span.type") == "inference")

    def input_of(self, span):
        return [json.loads(message) for message in span.events[0].attributes["input"]]

    def test_call_is_sent_as_written_when_no_instruction_is_in_effect(self):
        self.assertEqual(self.call_model(), "ok")
        self.assertEqual(self.sent_messages, DEVELOPER_MESSAGES)
        self.assertNotIn(SENT_INSTRUCTION, self.inference_span().attributes)
        self.assertEqual(self.input_of(self.inference_span()), [{"system": "You book travel."}, {"user": "A poem, please"}])

    def test_call_is_sent_with_the_instruction_first(self):
        os.environ.update(START_WITH_123456)
        self.call_model()
        self.assertEqual(self.sent_messages[0], {"role": "system", "content": instruction_in_effect().text})
        self.assertEqual(self.sent_messages[1:], DEVELOPER_MESSAGES)

    def test_span_input_lists_the_instruction_first_under_its_own_label(self):
        os.environ.update(START_WITH_123456)
        self.call_model()
        self.assertEqual(self.input_of(self.inference_span()), [
            {"instruction": instruction_in_effect().text}, {"system": "You book travel."}, {"user": "A poem, please"}])

    def test_call_that_arrives_already_instructed_lists_the_instruction_once(self):
        os.environ.update(START_WITH_123456)
        already_instructed = [{"role": "system", "content": instruction_in_effect().text}, *DEVELOPER_MESSAGES]
        wrapper = task_wrapper(self.tracer_provider.get_tracer("test"), SpanHandler(), CHAT_METHOD)
        wrapper(self.model_method, None, (), {"messages": already_instructed})
        self.assertEqual(self.sent_messages, already_instructed)
        self.assertEqual(self.input_of(self.inference_span()), [
            {"instruction": instruction_in_effect().text}, {"system": "You book travel."}, {"user": "A poem, please"}])

    def test_unsampled_call_is_still_sent_with_the_instruction(self):
        os.environ.update(START_WITH_123456)
        self.call_model(TracerProvider(sampler=ALWAYS_OFF))
        self.assertEqual(self.sent_messages[0]["content"], instruction_in_effect().text)

    def test_async_call_is_sent_with_the_instruction_and_its_span_names_it(self):
        os.environ.update(START_WITH_123456)
        wrapper = atask_wrapper(self.tracer_provider.get_tracer("test"), SpanHandler(), CHAT_METHOD)
        asyncio.run(wrapper(self.async_model_method, None, (), {"messages": DEVELOPER_MESSAGES}))
        self.assertEqual(self.sent_messages[0]["content"], instruction_in_effect().text)
        self.assertEqual(self.inference_span().attributes[SENT_INSTRUCTION], "start-with-123456")

    def test_call_with_spans_suppressed_is_still_sent_with_the_instruction(self):
        os.environ.update(START_WITH_123456)
        token = attach(set_value(MONOCLE_SUPPRESS_SPANS, True))
        try:
            self.call_model()
        finally:
            detach(token)
        self.assertEqual(self.sent_messages[0]["content"], instruction_in_effect().text)

    def test_only_the_span_of_the_model_call_names_the_instruction(self):
        os.environ.update(START_WITH_123456)
        self.call_model()
        spans_naming_it = [span for span in self.exporter.get_finished_spans() if SENT_INSTRUCTION in span.attributes]
        self.assertEqual([span.attributes["span.type"] for span in spans_naming_it], ["inference"])
        self.assertEqual(spans_naming_it[0].attributes[SENT_INSTRUCTION], "start-with-123456")

    def test_framework_span_lists_the_instruction_the_model_call_beneath_it_adds(self):
        os.environ.update(START_WITH_123456)
        framework_inference = {**CHAT_INFERENCE, "type": SPAN_TYPES.INFERENCE_FRAMEWORK}
        del framework_inference["add_instruction"]
        wrapper = task_wrapper(self.tracer_provider.get_tracer("test"), SpanHandler(), {**CHAT_METHOD, "output_processor": framework_inference})
        wrapper(self.model_method, None, (), {"messages": DEVELOPER_MESSAGES})
        self.assertEqual(self.sent_messages, DEVELOPER_MESSAGES)
        framework_span = next(span for span in self.exporter.get_finished_spans() if span.attributes.get("span.type") == "inference.framework")
        self.assertEqual(self.input_of(framework_span)[0], {"instruction": instruction_in_effect().text})

    def test_inference_span_that_adds_nothing_and_is_not_a_framework_is_not_labelled(self):
        os.environ.update(START_WITH_123456)
        plain_inference = {key: value for key, value in CHAT_INFERENCE.items() if key != "add_instruction"}
        wrapper = task_wrapper(self.tracer_provider.get_tracer("test"), SpanHandler(), {**CHAT_METHOD, "output_processor": plain_inference})
        wrapper(self.model_method, None, (), {"messages": DEVELOPER_MESSAGES})
        self.assertEqual(self.input_of(self.inference_span()), [{"system": "You book travel."}, {"user": "A poem, please"}])

    def test_failed_call_still_shows_the_instruction_it_was_sent_with(self):
        os.environ.update(START_WITH_123456)
        wrapper = task_wrapper(self.tracer_provider.get_tracer("test"), SpanHandler(), CHAT_METHOD)
        with self.assertRaises(RuntimeError):
            wrapper(self.failing_model_method, None, (), {"messages": DEVELOPER_MESSAGES})
        self.assertEqual(self.inference_span().attributes[SENT_INSTRUCTION], "start-with-123456")
        self.assertEqual(self.input_of(self.inference_span())[0], {"instruction": instruction_in_effect().text})

    def test_call_that_passes_its_messages_by_position_is_sent_as_written_and_not_labelled(self):
        os.environ.update(START_WITH_123456)
        positional_inference = {**CHAT_INFERENCE, "events": [{"name": "data.input", "attributes": [{
            "attribute": "input", "accessor": lambda arguments: [json.dumps({"user": "A poem, please"})]}]}]}
        wrapper = task_wrapper(self.tracer_provider.get_tracer("test"), SpanHandler(), {**CHAT_METHOD, "output_processor": positional_inference})
        wrapper(self.model_method, None, (DEVELOPER_MESSAGES,), {})
        self.assertEqual(self.sent_messages, DEVELOPER_MESSAGES)
        self.assertNotIn(SENT_INSTRUCTION, self.inference_span().attributes)
        self.assertEqual(self.input_of(self.inference_span()), [{"user": "A poem, please"}])

    def test_call_is_sent_as_written_when_the_instruction_file_is_missing(self):
        os.environ[MONOCLE_INSTRUCTION_ENV] = "config/house-style.yaml"
        with self.assertLogs(level="WARNING"):
            self.assertEqual(self.call_model(), "ok")
        self.assertEqual(self.sent_messages, DEVELOPER_MESSAGES)

    def test_input_reader_that_finds_no_messages_writes_no_input(self):
        no_messages_found = {"type": SPAN_TYPES.INFERENCE, "events": [{"name": "data.input", "attributes": [
            {"attribute": "input", "accessor": lambda arguments: []}]}]}
        wrapper = task_wrapper(self.tracer_provider.get_tracer("test"), SpanHandler(), {**CHAT_METHOD, "output_processor": no_messages_found})
        wrapper(self.model_method, None, (), {"messages": DEVELOPER_MESSAGES})
        self.assertNotIn("input", self.inference_span().events[0].attributes)

    def test_wrap_site_whose_output_processor_is_not_a_dict_is_sent_as_written(self):
        os.environ.update(START_WITH_123456)
        wrapper = task_wrapper(self.tracer_provider.get_tracer("test"), SpanHandler(), {**CHAT_METHOD, "output_processor": "named.elsewhere"})
        self.assertEqual(wrapper(self.model_method, None, (), {"messages": DEVELOPER_MESSAGES}), "ok")
        self.assertEqual(self.sent_messages, DEVELOPER_MESSAGES)

    def test_call_is_sent_as_written_when_the_instruction_cannot_be_placed(self):
        os.environ.update(START_WITH_123456)
        with patch.dict(CHAT_INFERENCE, {"add_instruction": lambda kwargs, text: 1 / 0}), self.assertLogs(level="WARNING"):
            self.assertEqual(self.call_model(), "ok")
        self.assertEqual(self.sent_messages, DEVELOPER_MESSAGES)
        self.assertNotIn(SENT_INSTRUCTION, self.inference_span().attributes)


if __name__ == "__main__":
    unittest.main()
