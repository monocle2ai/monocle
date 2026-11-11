"""Unit tests for the shared decision-model processor factory and helpers."""
import unittest
from dataclasses import replace

from opentelemetry.sdk.trace import TracerProvider

from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.decision._helper import (
    DecisionAdapter,
    decision_finish_reason,
    decision_finish_type,
    usage_metadata,
)
from monocle_apptrace.instrumentation.metamodel.decision.entities.inference import (
    make_decision_inference,
)


def _find_event(processor, name):
    return next(e for e in processor["events"] if e["name"] == name)


def _find_attr(attr_list, attribute):
    return next(a for a in attr_list if a.get("attribute") == attribute)


class TestDecisionHelpers(unittest.TestCase):
    def test_finish_reason_success_and_error(self):
        self.assertEqual(decision_finish_reason({"exception": None}), "success")
        self.assertEqual(decision_finish_reason({"exception": RuntimeError("x")}), "error")

    def test_finish_type_maps_through(self):
        self.assertEqual(decision_finish_type({"exception": None}), "success")
        self.assertEqual(decision_finish_type({"exception": RuntimeError("x")}), "error")

    def test_usage_metadata_sums_tokens(self):
        self.assertEqual(
            usage_metadata({"input_tokens": 120, "output_tokens": 5}),
            {"prompt_tokens": 120, "completion_tokens": 5, "total_tokens": 125},
        )

    def test_usage_metadata_empty(self):
        self.assertEqual(
            usage_metadata({}),
            {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        )


class TestDecisionProcessorFactory(unittest.TestCase):
    def setUp(self):
        self.adapter = DecisionAdapter(
            provider_name="demo",
            get_inference_endpoint=lambda instance: "https://demo.example",
            get_model=lambda arguments: "demo-1.0.0",
            extract_input=lambda arguments: ["in"],
            extract_answers=lambda arguments: "out",
            get_usage=lambda arguments: {"input_tokens": 1, "output_tokens": 2},
            model_type="decision",
        )
        self.processor = make_decision_inference(self.adapter)
        self.arguments = {"instance": object(), "args": (), "kwargs": {},
                          "result": object(), "exception": None}

    def test_span_type_is_inference(self):
        self.assertEqual(self.processor["type"], SPAN_TYPES.INFERENCE)

    def test_no_subtype_key(self):
        self.assertNotIn("subtype", self.processor)

    def test_provider_block(self):
        provider = self.processor["attributes"][0]
        self.assertEqual(_find_attr(provider, "type")["accessor"](self.arguments), "inference.demo")
        self.assertEqual(_find_attr(provider, "provider_name")["accessor"](self.arguments), "demo")
        self.assertEqual(
            _find_attr(provider, "inference_endpoint")["accessor"](self.arguments),
            "https://demo.example",
        )

    def test_model_block(self):
        model = self.processor["attributes"][1]
        self.assertEqual(_find_attr(model, "name")["accessor"](self.arguments), "demo-1.0.0")
        self.assertEqual(_find_attr(model, "type")["accessor"](self.arguments), "model.decision.demo-1.0.0")
        self.assertEqual(_find_attr(model, "model_type")["accessor"](self.arguments), "decision")

    def test_model_type_falls_back_to_completion(self):
        adapter = replace(self.adapter, model_type=None)
        model = make_decision_inference(adapter)["attributes"][1]
        self.assertEqual(_find_attr(model, "model_type")["accessor"](self.arguments), "completion")
        self.assertEqual(_find_attr(model, "type")["accessor"](self.arguments), "model.llm.demo-1.0.0")

    def test_model_block_is_post_execution(self):
        # entity.2 (model) reads result.model, so it MUST run in the post-execution pass.
        model = self.processor["attributes"][1]
        self.assertEqual(_find_attr(model, "name").get("phase"), "post_execution")
        self.assertEqual(_find_attr(model, "type").get("phase"), "post_execution")
        self.assertEqual(_find_attr(model, "model_type").get("phase"), "post_execution")

    def test_input_output_events(self):
        data_input = _find_event(self.processor, "data.input")
        self.assertEqual(_find_attr(data_input["attributes"], "input")["accessor"](self.arguments), ["in"])
        data_output = _find_event(self.processor, "data.output")
        self.assertEqual(_find_attr(data_output["attributes"], "response")["accessor"](self.arguments), "out")
        self.assertEqual(_find_attr(data_output["attributes"], "error_code")["accessor"](self.arguments), "")

    def test_metadata_event(self):
        metadata = _find_event(self.processor, "metadata")
        usage_attr = next(a for a in metadata["attributes"] if "attribute" not in a)
        self.assertEqual(
            usage_attr["accessor"](self.arguments),
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
        self.assertEqual(_find_attr(metadata["attributes"], "finish_reason")["accessor"](self.arguments), "success")
        self.assertEqual(_find_attr(metadata["attributes"], "finish_type")["accessor"](self.arguments), "success")


class TestDefaultModelType(unittest.TestCase):
    """LLM model entities get model_type=completion unless the processor sets one."""

    def _apply(self, attributes):
        tracer = TracerProvider().get_tracer(__name__)
        with tracer.start_as_current_span("test") as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
            SpanHandler.set_default_model_type(span, 2)
            return dict(span.attributes)

    def test_llm_model_defaults_to_completion(self):
        attrs = self._apply({"entity.1.type": "inference.openai", "entity.2.type": "model.llm.gpt-4o"})
        self.assertEqual(attrs["entity.2.model_type"], "completion")
        self.assertNotIn("entity.1.model_type", attrs)

    def test_existing_model_type_is_kept(self):
        attrs = self._apply({"entity.2.type": "model.llm.demo", "entity.2.model_type": "decision"})
        self.assertEqual(attrs["entity.2.model_type"], "decision")

    def test_non_llm_entities_are_skipped(self):
        attrs = self._apply({"entity.2.type": "model.embedding.text-embedding-3"})
        self.assertNotIn("entity.2.model_type", attrs)


if __name__ == "__main__":
    unittest.main()
