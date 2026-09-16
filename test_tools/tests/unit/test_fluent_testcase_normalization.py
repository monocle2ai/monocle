"""FluentTestCase must accept the JSON shapes real parametrized tests are written in.

Two of these shapes used to parse "successfully" while dropping the expectations
on the floor (the `expected` wrapper, an unknown key), which turns a broken test
case into a green test. That is the failure mode these tests exist to prevent.
"""
import pytest
from pydantic import ValidationError

from monocle_test_tools.schema import FactID
from monocle_test_tools.testcase import Agent, Eval, FluentTestCase, Tool


def test_expected_wrapper_is_lifted():
    tc = FluentTestCase.model_validate({
        "input": {"fact_id": "trace1232", "fact_name": "trace"},
        "expected": {"evals": {"hallucination": "minor_hallucination"}},
    })

    assert tc.input == FactID(fact_id="trace1232", fact_name="trace", source="file")
    assert tc.evals == [Eval(name="hallucination", result="minor_hallucination")]


def test_expected_wrapper_lifts_token_limit_too():
    tc = FluentTestCase.model_validate({"expected": {"token_limit": 10000}})

    assert tc.token_limit == 10000


def test_expected_key_colliding_with_top_level_raises():
    with pytest.raises(ValidationError, match="both inside and outside 'expected'"):
        FluentTestCase.model_validate({
            "evals": {"a": "x"},
            "expected": {"evals": {"b": "y"}},
        })


def test_evals_mapping_becomes_a_list_in_order():
    tc = FluentTestCase.model_validate({
        "evals": {"hallucination": "minor", "frustration": "none"},
    })

    assert tc.evals == [Eval(name="hallucination", result="minor"),
                        Eval(name="frustration", result="none")]


def test_agents_mapping_becomes_a_list():
    tc = FluentTestCase.model_validate({
        "agents": {"supervisor": {}, "flight_agent": {"output": "booked"}},
    })

    assert tc.agents == [Agent(name="supervisor"),
                         Agent(name="flight_agent", output="booked")]


def test_tools_mapping_becomes_a_list():
    tc = FluentTestCase.model_validate({"tools": {"book_flight": {"output": "ok"}}})

    assert tc.tools == [Tool(name="book_flight", output="ok")]


def test_single_flat_entry_dict_is_wrapped_not_split():
    """A dict whose keys are the model's own fields is one flat entry, not a mapping."""
    tc = FluentTestCase.model_validate({
        "agents": {"name": "supervisor", "output": "booked"},
    })

    assert tc.agents == [Agent(name="supervisor", output="booked")]


def test_evals_list_form_still_works():
    tc = FluentTestCase.model_validate({"evals": [{"hallucination": "minor"}]})

    assert tc.evals == [Eval(name="hallucination", result="minor")]


def test_scalar_input_becomes_a_one_tuple():
    tc = FluentTestCase.model_validate({"input": "Book a flight"})

    assert tc.input == ("Book a flight",)


def test_tuple_input_is_untouched():
    tc = FluentTestCase.model_validate({"input": ("a", "b")})

    assert tc.input == ("a", "b")


def test_factid_input_is_untouched():
    tc = FluentTestCase.model_validate({"input": {"fact_id": "t1", "source": "okahu"}})

    assert tc.input == FactID(fact_id="t1", fact_name="traces", source="okahu")


def test_unknown_key_raises():
    with pytest.raises(ValidationError):
        FluentTestCase.model_validate({"evels": [{"hallucination": "minor"}]})


def test_unknown_key_inside_expected_raises():
    with pytest.raises(ValidationError):
        FluentTestCase.model_validate({"expected": {"evels": [{"h": "m"}]}})


def test_round_trips_through_dump():
    """from_spans -> dump -> re-parse must still produce an equal model."""
    original = FluentTestCase.model_validate({
        "name": "t", "input": ("go",),
        "agents": {"supervisor": {"output": "done"}},
        "tools": {"book": {"output": "ok"}},
        "evals": {"hallucination": "minor"},
        "token_limit": 42,
    })

    assert FluentTestCase.model_validate(original.model_dump()) == original


class TestEvalCategory:
    """An eval may name a category, written into the key as "<name>@<category>"."""

    def test_a_categorised_key_splits(self):
        assert Eval.model_validate({"hallucination@llm": "minor"}) == Eval(
            name="hallucination", category="llm", result="minor")

    def test_an_uncategorised_key_is_unchanged(self):
        tc = Eval.model_validate({"hallucination": "minor"})

        assert tc.name == "hallucination"
        assert tc.category is None

    def test_a_bare_categorised_string(self):
        assert Eval.model_validate("hallucination@manual") == Eval(
            name="hallucination", category="manual")

    def test_the_flat_form_still_works(self):
        assert Eval.model_validate(
            {"name": "hallucination", "category": "llm", "result": "minor"}
        ) == Eval(name="hallucination", category="llm", result="minor")

    def test_serializes_with_the_category_in_the_key(self):
        assert Eval(name="hallucination", category="llm",
                    result="minor").model_dump() == {"hallucination@llm": "minor"}

    def test_serializes_without_one_when_absent(self):
        assert Eval(name="hallucination",
                    result="minor").model_dump() == {"hallucination": "minor"}

    @pytest.mark.parametrize("payload", [
        {"hallucination@llm": "minor"},
        {"hallucination": "minor"},
    ])
    def test_round_trips(self, payload):
        once = Eval.model_validate(payload)

        assert Eval.model_validate(once.model_dump()) == once

    def test_only_the_last_at_separates(self):
        """An eval name may itself contain @; the category is what follows the last."""
        assert Eval.model_validate({"a@b@llm": "minor"}) == Eval(
            name="a@b", category="llm", result="minor")

    def test_an_empty_category_is_no_category(self):
        tc = Eval.model_validate({"hallucination@": "minor"})

        assert tc.name == "hallucination"
        assert tc.category is None

    def test_a_path_name_is_not_split(self):
        """A custom-template path is a name, not a name@category pair."""
        from pathlib import Path

        tc = Eval.model_validate(Path("./evals/my@eval.json"))

        assert tc.name == Path("./evals/my@eval.json")
        assert tc.category is None

    def test_works_through_the_testcase_mapping_form(self):
        tc = FluentTestCase.model_validate({
            "evals": {"hallucination@llm": "minor", "bias@manual": "biased"}})

        assert tc.evals == [Eval(name="hallucination", category="llm", result="minor"),
                            Eval(name="bias", category="manual", result="biased")]
