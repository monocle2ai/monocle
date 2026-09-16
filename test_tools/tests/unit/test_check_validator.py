"""check_validator: the test author's own pass/fail logic over recorded responses.

Nothing here grades against an expected label -- the validator decides. So these
tests cover that contract: what a validator may return, which spans it sees, how
it can be named, and how rejections are reported.
"""
import importlib
from types import SimpleNamespace

import pytest

from monocle_test_tools import BaseValidator
from monocle_test_tools.custom_validator import get_validator, read_verdict
from monocle_test_tools.fluent_api import TraceAssertion
from validator_examples import MinimumLengthValidator


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    TraceAssertion._assertion_errors = []
    TraceAssertion._testcase_mode = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._testcase_mode = None


def _span(name, input=None, output=None):
    """A span carrying the data.input/data.output events trace_utils reads."""
    events = []
    if input is not None:
        events.append(SimpleNamespace(name="data.input", attributes={"input": input}))
    if output is not None:
        events.append(SimpleNamespace(name="data.output", attributes={"response": output}))
    return SimpleNamespace(name=name, events=events, attributes={})


def _asserter(*spans):
    return TraceAssertion(filtered_spans=list(spans) or [_span("agent", "ask", "answer")])


def _drain():
    messages = [a["message"] for a in TraceAssertion._assertion_errors]
    TraceAssertion._assertion_errors = []
    return messages


# --- the verdict contract -------------------------------------------------

def test_true_passes():
    _asserter().check_validator(lambda input, output: True)

    assert _drain() == []


def test_false_fails_with_a_generated_message():
    def always_bad(input, output):
        return False

    _asserter().check_validator(always_bad)

    assert _drain() == ["agent: validator 'always_bad' rejected the response"]


def test_a_returned_string_becomes_the_failure_message():
    def needs_ref(input, output):
        return f"no booking reference in {output!r}"

    _asserter().check_validator(needs_ref)

    assert _drain() == ["agent: no booking reference in 'answer'"]


def test_an_empty_string_still_reads_as_a_rejection():
    _asserter().check_validator(lambda input, output: "   ")

    assert _drain() == ["agent: validator '<lambda>' rejected the response"]


def test_returning_none_is_an_error_not_a_pass():
    def forgot_to_return(input, output):
        output.strip()

    with pytest.raises(TypeError, match="returned None"):
        _asserter().check_validator(forgot_to_return)


def test_returning_something_outside_the_contract_is_an_error():
    with pytest.raises(TypeError, match="returned dict"):
        _asserter().check_validator(lambda input, output: {"valid": 1.0})


def test_a_truthy_non_bool_is_not_silently_a_pass():
    with pytest.raises(TypeError, match="returned int"):
        _asserter().check_validator(lambda input, output: 1)


# --- what the validator is handed ----------------------------------------

def test_the_recorded_input_and_output_are_passed():
    seen = []

    def record(input, output):
        seen.append((input, output))
        return True

    _asserter(_span("agent", "book a flight", "booked")).check_validator(record)

    assert seen == [("book a flight", "booked")]


def test_every_selected_span_is_validated():
    seen = []

    def record(input, output):
        seen.append(output)
        return True

    _asserter(_span("a", "i", "one"), _span("b", "i", "two")).check_validator(record)

    assert seen == ["one", "two"]


def test_spans_recording_nothing_are_left_out():
    seen = []

    def record(input, output):
        seen.append(output)
        return True

    _asserter(_span("workflow"), _span("agent", "i", "answer")).check_validator(record)

    assert seen == ["answer"]
    assert _drain() == []


def test_nothing_to_validate_is_a_failure_not_a_pass():
    _asserter(_span("workflow")).check_validator(lambda input, output: True)

    assert _drain() == [
        "No spans with a recorded input or output to validate. Chain a span "
        "selector before check_validator."]


def test_a_span_with_only_an_output_is_still_validated():
    _asserter(_span("agent", output="answer")).check_validator(
        lambda input, output: input is None and output == "answer")

    assert _drain() == []


# --- reporting ------------------------------------------------------------

def test_every_rejection_is_reported_together():
    _asserter(_span("a", "i", "one"), _span("b", "i", "two")).check_validator(
        lambda input, output: f"bad {output}")

    assert _drain() == [
        "Validation '<lambda>' failed for 2 of 2 checks:"
        "\n  - a: bad one"
        "\n  - b: bad two"]


def test_a_custom_message_replaces_the_generated_one():
    _asserter().check_validator(lambda input, output: False, message="agent went off script")

    assert _drain() == ["agent went off script"]


def test_passing_spans_are_not_reported():
    _asserter(_span("a", "i", "good"), _span("b", "i", "bad")).check_validator(
        lambda input, output: True if output == "good" else "not good")

    assert _drain() == ["b: not good"]


# --- ways of naming a validator -------------------------------------------

def test_a_base_validator_subclass_is_accepted():
    class RejectsEverything(BaseValidator):
        def validate_response(self, input=None, output=None):
            return False

    _asserter().check_validator(RejectsEverything())

    assert _drain() == ["agent: validator 'RejectsEverything' rejected the response"]


def test_a_base_validator_class_is_instantiated():
    class AcceptsEverything(BaseValidator):
        def validate_response(self, input=None, output=None):
            return True

    _asserter().check_validator(AcceptsEverything)

    assert _drain() == []


def test_validator_options_reach_the_instance():
    _asserter().check_validator(MinimumLengthValidator(validator_options={"minimum": 50}))

    assert _drain() == ["agent: output shorter than 50 characters"]


def test_an_import_path_is_resolved():
    _asserter().check_validator("validator_examples:rejects_all")

    assert _drain() == ["agent: rejected by import-path validator"]


def test_the_dotted_import_path_spelling_also_works():
    _asserter().check_validator("validator_examples.accepts_all")

    assert _drain() == []


def test_an_unimportable_module_names_the_reference():
    with pytest.raises(ValueError, match="cannot import module 'no_such_module'"):
        _asserter().check_validator("no_such_module:check")


def test_a_missing_attribute_names_the_reference():
    with pytest.raises(ValueError, match="has no 'no_such_check'"):
        _asserter().check_validator("validator_examples:no_such_check")


def test_a_reference_with_no_module_is_rejected():
    with pytest.raises(ValueError, match="does not name an importable validator"):
        _asserter().check_validator("valid_order")


def test_something_that_is_not_a_validator_at_all_is_rejected():
    with pytest.raises(TypeError, match="must be a function, a BaseValidator"):
        _asserter().check_validator("validator_examples:not_callable")


def test_a_validator_with_the_wrong_signature_is_caught_up_front():
    def wrong_names(prompt, response):
        return True

    with pytest.raises(ValueError, match="must be callable as func"):
        _asserter().check_validator(wrong_names)


def test_a_validator_taking_kwargs_is_accepted():
    _asserter().check_validator(lambda **kwargs: True)

    assert _drain() == []


# --- testcase-driven ------------------------------------------------------

def test_a_testcase_runs_its_validators():
    _asserter().check_validator(
        testcase={"validators": ["validator_examples:accepts_all"]})

    assert _drain() == []


def test_a_testcase_runs_every_validator_it_names():
    calls = []

    def first(input, output):
        calls.append("first")
        return "first says no"

    def second(input, output):
        calls.append("second")
        return "second says no"

    _asserter().check_validator(testcase={"name": "booking", "validators": [first, second]})

    assert calls == ["first", "second"]
    assert _drain() == [
        "Validation 'booking' failed for 2 of 2 checks:"
        "\n  - agent: first says no"
        "\n  - agent: second says no"]


def test_a_single_validator_need_not_be_a_list():
    _asserter().check_validator(testcase={"validators": "validator_examples:rejects_all"})

    assert _drain() == ["agent: rejected by import-path validator"]


def test_a_testcase_with_no_validators_is_not_a_passing_test():
    with pytest.raises(ValueError, match="has no validators to run"):
        _asserter().check_validator(testcase={"input": "go"})


def test_a_testcase_cannot_be_combined_with_a_validator():
    with pytest.raises(ValueError, match="cannot be combined with 'testcase'"):
        _asserter().check_validator(lambda input, output: True,
                                    testcase={"validators": ["validator_examples:accepts_all"]})


def test_a_validator_is_required_without_a_testcase():
    with pytest.raises(ValueError, match="'validator' is required"):
        _asserter().check_validator()


# --- unit-level contract helpers -----------------------------------------

def test_read_verdict_maps_the_contract():
    assert read_verdict(True, name="v") is None
    assert read_verdict(False, name="v") == "validator 'v' rejected the response"
    assert read_verdict("why", name="v") == "why"


def test_get_validator_returns_a_base_validator_instance():
    resolved = get_validator(lambda input, output: True)

    assert isinstance(resolved, BaseValidator)
    assert resolved.validate_response(input="i", output="o") is True


# --- when the validator itself misbehaves ---------------------------------

def test_an_assert_style_validator_is_reported_as_a_rejection():
    """A validator written with `assert` still reads as a rejection, with its span."""
    def asserts_instead_of_returning(input, output):
        assert "ORDER-" in output, "no booking reference"

    _asserter().check_validator(asserts_instead_of_returning)

    # pytest's assertion rewriting appends the expression it introspected, so match
    # the part we produce rather than pinning pytest's formatting.
    assert _drain()[0].startswith("agent: no booking reference")


def test_an_assert_style_validator_does_not_stop_the_other_spans():
    """The all-failures-together guarantee survives an assert-style validator."""
    def asserts(input, output):
        assert False, f"rejected {output}"

    _asserter(_span("a", "i", "one"), _span("b", "i", "two")).check_validator(asserts)

    message = _drain()[0]
    assert message.startswith("Validation 'asserts' failed for 2 of 2 checks:")
    assert "\n  - a: rejected one" in message
    assert "\n  - b: rejected two" in message


def test_a_crashing_validator_names_the_validator_and_the_span():
    """A bug in the validator is an error, and says which span it died on."""
    def crashes(input, output):
        return "ORDER-" in output.lower()

    # The second span recorded no output, which is how this usually happens.
    with pytest.raises(RuntimeError, match=r"validator 'crashes' raised on span 'b': AttributeError"):
        _asserter(_span("a", "i", "one"), _span("b", "i")).check_validator(crashes)


def test_an_async_validator_is_refused_where_the_test_names_it():
    """Async validators would only ever return a coroutine, so they are caught early."""
    async def checks_later(input, output):
        return True

    with pytest.raises(ValueError, match="is async"):
        _asserter().check_validator(checks_later)


def test_an_async_base_validator_is_refused_too():
    class AsyncValidator(BaseValidator):
        async def validate_response(self, input=None, output=None):
            return True

    with pytest.raises(ValueError, match="async validate_response"):
        _asserter().check_validator(AsyncValidator())


def test_a_module_that_fails_at_import_names_the_reference(monkeypatch):
    """A module that is found but blows up while executing still names the reference."""
    def explode(name):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(importlib, "import_module", explode)

    with pytest.raises(ValueError, match="failed: ZeroDivisionError"):
        _asserter().check_validator("a_broken_module:check")
