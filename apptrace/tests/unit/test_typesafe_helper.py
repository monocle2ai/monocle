"""Unit tests for the typesafe (jev) decision-model adapter helpers.

Uses SimpleNamespace fakes shaped like typesafe_sdk's SystemOneResponse, since the
real SDK is not installed in the monocle environment.
"""
import json
import unittest
from types import SimpleNamespace

from monocle_apptrace.instrumentation.metamodel.typesafe import _helper


def _choice(choice, confidence, probabilities):
    return SimpleNamespace(type="choice", choice=choice, confidence=confidence,
                           probabilities=probabilities)


def _noul(noul):
    return SimpleNamespace(type="noul", noul=noul)


def _score(score, confidence, probabilities, legend):
    return SimpleNamespace(type="score", score=score, confidence=confidence,
                           probabilities=probabilities, legend=legend)


def _response(model=None, input_tokens=None, output_tokens=None, answers=None):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        answers=answers or {},
    )


def _args(instance=None, args=(), kwargs=None, result=None, exception=None):
    return {"instance": instance or SimpleNamespace(), "args": args,
            "kwargs": kwargs or {}, "result": result, "exception": exception}


class TestGetModel(unittest.TestCase):
    def test_prefers_resolved_model(self):
        arguments = _args(kwargs={"model": "jev-latest"}, result=_response(model="jev-1.13.0"))
        self.assertEqual(_helper.get_model(arguments), "jev-1.13.0")

    def test_falls_back_to_requested_alias(self):
        arguments = _args(kwargs={"model": "jev-latest"}, result=_response(model=None))
        self.assertEqual(_helper.get_model(arguments), "jev-latest")

    def test_none_when_absent(self):
        self.assertIsNone(_helper.get_model(_args(result=_response())))


class TestEndpoint(unittest.TestCase):
    def test_reads_base_url(self):
        self.assertEqual(
            _helper.get_inference_endpoint(SimpleNamespace(base_url="https://api.typesafe.ai")),
            "https://api.typesafe.ai",
        )

    def test_none_when_missing(self):
        self.assertIsNone(_helper.get_inference_endpoint(SimpleNamespace()))


class TestExtractInput(unittest.TestCase):
    def test_kwargs_state_and_questions(self):
        question = SimpleNamespace(type="choice", instructions="What is the tone?",
                                   criteria={"calm": None, "angry": None})
        arguments = _args(kwargs={"state": "I was charged twice.",
                                  "questions": {"label": question}})
        out = _helper.extract_input(arguments)
        self.assertEqual(json.loads(out[0]), {"state": "I was charged twice."})
        self.assertEqual(
            json.loads(out[1]),
            {"label": {"type": "choice", "instructions": "What is the tone?",
                       "criteria": {"calm": None, "angry": None}}},
        )

    def test_positional_state_and_questions(self):
        question = SimpleNamespace(type="noul", instructions="Is this spam?", criteria=None)
        arguments = _args(args=("some text", {"spam": question}))
        out = _helper.extract_input(arguments)
        self.assertEqual(json.loads(out[0]), {"state": "some text"})
        self.assertEqual(json.loads(out[1]), {"spam": {"type": "noul", "instructions": "Is this spam?"}})


class TestExtractAnswers(unittest.TestCase):
    def test_choice_answer(self):
        result = _response(answers={"label": _choice("angry", 0.9, {"angry": 0.8, "calm": 0.2})})
        out = json.loads(_helper.extract_answers(_args(result=result)))
        self.assertEqual(out["label"], {"type": "choice", "choice": "angry",
                                        "confidence": 0.9,
                                        "probabilities": {"angry": 0.8, "calm": 0.2}})

    def test_mixed_answer_types(self):
        result = _response(answers={
            "label": _choice("angry", 0.9, {"angry": 0.8, "calm": 0.2}),
            "spam": _noul(0.98),
            "urgency": _score(1.7, 0.8, {"0": 0.1, "1": 0.1, "2": 0.8}, {"0": "low"}),
        })
        out = json.loads(_helper.extract_answers(_args(result=result)))
        self.assertEqual(out["spam"], {"type": "noul", "noul": 0.98})
        self.assertEqual(out["urgency"]["type"], "score")
        self.assertEqual(out["urgency"]["score"], 1.7)
        self.assertEqual(out["label"]["choice"], "angry")

    def test_empty_answers(self):
        self.assertEqual(_helper.extract_answers(_args(result=_response())), "")


class TestGetUsage(unittest.TestCase):
    def test_reads_tokens(self):
        arguments = _args(result=_response(input_tokens=120, output_tokens=5))
        self.assertEqual(_helper.get_usage(arguments), {"input_tokens": 120, "output_tokens": 5})

    def test_no_usage(self):
        self.assertEqual(_helper.get_usage(_args(result=SimpleNamespace(usage=None))), {})


class TestAdapter(unittest.TestCase):
    def test_adapter_fields_wired(self):
        self.assertEqual(_helper.TYPESAFE_ADAPTER.provider_name, "typesafe")
        self.assertIs(_helper.TYPESAFE_ADAPTER.get_model, _helper.get_model)
        self.assertIs(_helper.TYPESAFE_ADAPTER.extract_answers, _helper.extract_answers)


if __name__ == "__main__":
    unittest.main()
