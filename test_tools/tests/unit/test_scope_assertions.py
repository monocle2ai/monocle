import os
import pytest
from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.validator import MonocleValidator
from monocle_test_tools.comparer.default_comparer import DefaultComparer
from monocle_test_tools.comparer.token_match_comparer import TokenMatchComparer
from span_loader import JSONSpanLoader

# Scope present on every span in trace1.json (span_handler writes "scope.<name>").
SCOPE_NAME = "agentic.request"
SCOPE_VALUE = "0x70adde2fc9b7f98ddbda564962a6028"


@pytest.fixture(scope="module")
def spans():
    """Load spans that carry a known scope attribute."""
    current_script_path = os.path.abspath(__file__)
    return JSONSpanLoader.from_json(
        os.path.join(os.path.dirname(current_script_path), "traces/trace1.json")
    )


@pytest.fixture
def validator():
    return MonocleValidator()


@pytest.fixture
def asserter(spans):
    """Fresh TraceAssertion seeded with spans, with shared assertion state reset.

    The reset matters: a pytest hook in the plugin flips any passing test to
    failed when TraceAssertion._assertion_errors is non-empty.
    """
    TraceAssertion._assertion_errors = []
    yield TraceAssertion(filtered_spans=spans)
    TraceAssertion._assertion_errors = []


def test_check_scope_exact_match(validator, spans):
    matched = validator._check_scope(spans, SCOPE_NAME, [SCOPE_VALUE], DefaultComparer())
    assert len(matched) == len(spans)  # every span carries this scope


def test_check_scope_wrong_value_no_match(validator, spans):
    matched = validator._check_scope(spans, SCOPE_NAME, ["nope"], DefaultComparer())
    assert matched == []


def test_check_scope_unknown_name_no_match(validator, spans):
    matched = validator._check_scope(spans, "no_such_scope", [SCOPE_VALUE], DefaultComparer())
    assert matched == []


def test_check_scope_any_of_values(validator, spans):
    matched = validator._check_scope(spans, SCOPE_NAME, ["wrong", SCOPE_VALUE], DefaultComparer())
    assert len(matched) == len(spans)


def test_check_scope_no_duplicate_span_when_multiple_values_match(validator, spans):
    matched = validator._check_scope(spans, SCOPE_NAME, [SCOPE_VALUE, SCOPE_VALUE], DefaultComparer())
    assert len(matched) == len(spans)


def test_check_scope_existence_only(validator, spans):
    matched = validator._check_scope(spans, SCOPE_NAME, None, DefaultComparer())
    assert len(matched) == len(spans)


def test_check_scope_existence_only_absent(validator, spans):
    matched = validator._check_scope(spans, "no_such_scope", None, DefaultComparer())
    assert matched == []


def test_check_scope_substring(validator, spans):
    matched = validator._check_scope(spans, SCOPE_NAME, ["adde2fc"], TokenMatchComparer())
    assert len(matched) == len(spans)


def test_check_scope_substring_case_insensitive(validator, spans):
    matched = validator._check_scope(spans, SCOPE_NAME, ["ADDE2FC"], TokenMatchComparer())
    assert len(matched) == len(spans)


def test_check_scope_substring_no_match(validator, spans):
    matched = validator._check_scope(spans, SCOPE_NAME, ["zzz"], TokenMatchComparer())
    assert matched == []



def _verify(asserter, scope_name, values, comparer, positive_test):
    """Invoke the verification helper directly, bypassing the assertion-collecting
    decorator so we can observe AssertionError instead of recorded state."""
    asserter._verify_scope(asserter._filtered_spans, scope_name, values,
                           comparer=comparer, positive_test=positive_test)


def test_verify_positive_match_does_not_raise(asserter):
    _verify(asserter, SCOPE_NAME, [SCOPE_VALUE], DefaultComparer(), True)


def test_verify_positive_no_match_raises(asserter):
    with pytest.raises(AssertionError):
        _verify(asserter, SCOPE_NAME, ["nope"], DefaultComparer(), True)


def test_verify_negative_absent_does_not_raise(asserter):
    _verify(asserter, SCOPE_NAME, ["nope"], DefaultComparer(), False)


def test_verify_negative_present_raises(asserter):
    with pytest.raises(AssertionError):
        _verify(asserter, SCOPE_NAME, [SCOPE_VALUE], DefaultComparer(), False)


def test_verify_existence_present_does_not_raise(asserter):
    _verify(asserter, SCOPE_NAME, None, DefaultComparer(), True)


def test_verify_existence_absent_raises(asserter):
    with pytest.raises(AssertionError):
        _verify(asserter, "no_such_scope", None, DefaultComparer(), True)


# --- public fluent API: happy-path wiring + input validation --------------
# Only assert success here; recording a failure via the decorated methods would
# pollute TraceAssertion._assertion_errors and trip the plugin's report hook.

def test_has_scope_match_passes(asserter):
    assert not asserter.has_scope(SCOPE_NAME, SCOPE_VALUE).is_assertion_failed


def test_has_any_scope_match_passes(asserter):
    assert not asserter.has_any_scope(SCOPE_NAME, "wrong", SCOPE_VALUE).is_assertion_failed


def test_contains_scope_match_passes(asserter):
    assert not asserter.contains_scope(SCOPE_NAME, "adde2fc").is_assertion_failed


def test_does_not_have_scope_absent_passes(asserter):
    assert not asserter.does_not_have_scope(SCOPE_NAME, "nope").is_assertion_failed


def test_has_scope_existence_present_passes(asserter):
    assert not asserter.has_scope(SCOPE_NAME).is_assertion_failed


def test_does_not_have_scope_absence_passes(asserter):
    assert not asserter.does_not_have_scope("no_such_scope").is_assertion_failed


@pytest.mark.parametrize("method", [
    "has_any_scope", "does_not_have_any_scope",
    "contains_any_scope", "does_not_contain_any_scope",
])
def test_any_scope_methods_require_a_value(asserter, method):
    with pytest.raises(ValueError):
        getattr(asserter, method)(SCOPE_NAME)


if __name__ == "__main__":
    pytest.main([__file__])
