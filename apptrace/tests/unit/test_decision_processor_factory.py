"""Unit tests for the shared decision-model processor factory and helpers."""
import unittest

from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
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


if __name__ == "__main__":
    unittest.main()
