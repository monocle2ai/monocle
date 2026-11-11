"""End-to-end: a fake TypeSafe-style client, instrumented with the decision processor,
produces an INFERENCE span with the expected entity attributes and events.

No typesafe_sdk required — the fake client is registered via wrapper_methods at this
module's import path.
"""
import asyncio
import json
import unittest
from types import SimpleNamespace

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from monocle_apptrace.instrumentation.common.instrumentor import (
    get_monocle_instrumentor,
    setup_monocle_telemetry,
)
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.typesafe.methods import DECISION_INFERENCE


class FakeChoiceAnswer:
    type = "choice"
    def __init__(self, choice, confidence, probabilities):
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probabilities


class FakeSystemOneResponse:
    def __init__(self, model, input_tokens, output_tokens, answers):
        self.model = model
        self.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
        self.answers = answers


class FakeTypeSafeClient:
    base_url = "https://api.typesafe.ai"
    def system_one(self, state, questions, *, model=None):
        return FakeSystemOneResponse(
            model="jev-1.13.0",
            input_tokens=120,
            output_tokens=5,
            answers={"label": FakeChoiceAnswer("angry", 0.9, {"angry": 0.8, "calm": 0.2})},
        )


class FakeAsyncTypeSafeClient:
    base_url = "https://api.typesafe.ai"
    async def system_one(self, state, questions, *, model=None):
        return FakeSystemOneResponse(
            model="jev-1.13.0",
            input_tokens=120,
            output_tokens=5,
            answers={"label": FakeChoiceAnswer("angry", 0.9, {"angry": 0.8, "calm": 0.2})},
        )


def _attr(span, key):
    return span.attributes.get(key)


class TestTypesafeInstrumentation(unittest.TestCase):
    def setUp(self):
        existing = get_monocle_instrumentor()
        if existing is not None:
            try:
                existing.uninstrument()
            except Exception:
                pass
        self.exporter = InMemorySpanExporter()
        self.instrumentor = setup_monocle_telemetry(
            workflow_name="typesafe_test",
            span_processors=[SimpleSpanProcessor(self.exporter)],
            wrapper_methods=[
                {
                    "package": __name__,
                    "object": "FakeTypeSafeClient",
                    "method": "system_one",
                    "wrapper_method": task_wrapper,
                    "output_processor": DECISION_INFERENCE,
                    "span_handler": "non_framework_handler",
                },
                {
                    "package": __name__,
                    "object": "FakeAsyncTypeSafeClient",
                    "method": "system_one",
                    "wrapper_method": atask_wrapper,
                    "output_processor": DECISION_INFERENCE,
                    "span_handler": "non_framework_handler",
                },
            ],
            union_with_default_methods=False,
        )

    def tearDown(self):
        try:
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            print("Uninstrument failed:", e)

    def _run(self):
        FakeTypeSafeClient().system_one(
            state="I was charged twice.",
            questions={"label": SimpleNamespace(type="choice", instructions="Tone?",
                                                criteria={"calm": None, "angry": None})},
            model="jev-latest",
        )
        spans = self.exporter.get_finished_spans()
        inference = [s for s in spans if s.attributes.get("span.type") == "inference"]
        self.assertTrue(inference, f"no inference span; span types: "
                                   f"{[s.attributes.get('span.type') for s in spans]}")
        return inference[0]

    def test_entity_attributes(self):
        span = self._run()
        self.assertEqual(_attr(span, "entity.1.type"), "inference.typesafe")
        self.assertEqual(_attr(span, "entity.1.provider_name"), "typesafe")
        self.assertEqual(_attr(span, "entity.1.inference_endpoint"), "https://api.typesafe.ai")
        # Resolved model, not the requested "jev-latest" alias.
        self.assertEqual(_attr(span, "entity.2.name"), "jev-1.13.0")
        self.assertEqual(_attr(span, "entity.2.type"), "model.decision.jev-1.13.0")
        self.assertEqual(_attr(span, "entity.2.model_type"), "decision")
        # Model is the 2nd entity; lock the count so a regression where the model block
        # fails to register as entity.2 (the bug fixed on this branch) is caught.
        self.assertEqual(_attr(span, "entity.count"), 2)

    def test_events(self):
        span = self._run()
        events = {e.name: dict(e.attributes) for e in span.events}
        self.assertIn("data.input", events)
        self.assertIn("data.output", events)
        self.assertIn("metadata", events)

        response = json.loads(events["data.output"]["response"])
        self.assertEqual(response["label"]["choice"], "angry")
        self.assertEqual(response["label"]["confidence"], 0.9)
        self.assertEqual(response["label"]["probabilities"], {"angry": 0.8, "calm": 0.2})

        meta = events["metadata"]
        self.assertEqual(meta["prompt_tokens"], 120)
        self.assertEqual(meta["completion_tokens"], 5)
        self.assertEqual(meta["total_tokens"], 125)
        self.assertEqual(meta["finish_reason"], "success")
        self.assertEqual(meta["finish_type"], "success")

    def test_async_resolved_model_and_events(self):
        asyncio.run(FakeAsyncTypeSafeClient().system_one(
            state="I was charged twice.",
            questions={"label": SimpleNamespace(type="choice", instructions="Tone?",
                                                criteria={"calm": None, "angry": None})},
            model="jev-latest",
        ))
        spans = self.exporter.get_finished_spans()
        inference = [s for s in spans if s.attributes.get("span.type") == "inference"]
        self.assertTrue(inference, f"no inference span; got "
                                   f"{[s.attributes.get('span.type') for s in spans]}")
        span = inference[0]
        # Resolved model via the async path, not the requested alias.
        self.assertEqual(_attr(span, "entity.2.name"), "jev-1.13.0")
        self.assertEqual(_attr(span, "entity.2.type"), "model.decision.jev-1.13.0")
        self.assertEqual(_attr(span, "entity.2.model_type"), "decision")
        self.assertEqual(_attr(span, "entity.count"), 2)
        events = {e.name: dict(e.attributes) for e in span.events}
        response = json.loads(events["data.output"]["response"])
        self.assertEqual(response["label"]["choice"], "angry")
        meta = events["metadata"]
        self.assertEqual(meta["total_tokens"], 125)
        self.assertEqual(meta["finish_type"], "success")


if __name__ == "__main__":
    unittest.main()
