"""Argument resolution shared by the fluent entry points that accept ``testcase=``.

These are pure functions over a ``FluentTestCase`` with no ``TraceAssertion``
dependency, so they stay testable on their own and are reusable by the span
selectors when those gain ``testcase=`` support.
"""
from typing import Any, Optional, Sequence, Union

from opentelemetry.sdk.trace import ReadableSpan

from monocle_test_tools.schema import FactID, SpanType
from monocle_test_tools.testcase import TURN_SPAN_TYPES, FluentTestCase
from monocle_test_tools.trace_utils import get_input_from_span


def _reject_unloadable(testcase: FluentTestCase) -> FluentTestCase:
    """Fail the test when the case's data could not be loaded.

    setup_test_cases emits a case per fact even when that fact's spans or evals
    could not be fetched, so the gap is reported rather than silently shrinking
    the suite. Every ``testcase=`` entry point resolves through here, so this is
    the one place the recorded error has to surface.

    AssertionError rather than ValueError: nothing is wrong with how the test was
    written, so it belongs in the failure column with the other assertions.
    """
    if testcase.load_error:
        raise AssertionError(
            f"testcase '{testcase.name}' could not be loaded, so it cannot be "
            f"asserted on: {testcase.load_error}")
    return testcase


def resolve_testcase(testcase: Union[FluentTestCase, dict], **forbidden: Any) -> FluentTestCase:
    """Normalize *testcase* to a model, rejecting arguments it conflicts with.

    A test case already carries the values the *forbidden* arguments would
    supply, so a caller passing both has written a test whose intent is
    ambiguous -- that is a mistake in the test, not a case to silently resolve
    by precedence.

    Args:
        testcase: A FluentTestCase, or a dict in any shape it accepts.
        **forbidden: Arguments that must not accompany a test case, by name. An
            argument counts as given when it is neither None nor an empty tuple,
            so a caller can forward its ``*args`` directly.

    Returns:
        The test case as a FluentTestCase.

    Raises:
        ValueError: If any forbidden argument was given.
        TypeError: If *testcase* is neither a FluentTestCase nor a dict.
    """
    given = [name for name, value in forbidden.items()
             if value is not None and value != ()]
    if given:
        names = ", ".join(f"'{name}'" for name in sorted(given))
        raise ValueError(
            f"{names} cannot be combined with 'testcase'; the test case already "
            "supplies these values")

    if isinstance(testcase, FluentTestCase):
        return _reject_unloadable(testcase)
    if isinstance(testcase, dict):
        return _reject_unloadable(FluentTestCase.model_validate(testcase))
    raise TypeError(
        f"testcase must be a FluentTestCase or dict, got {type(testcase).__name__}")


# FactID.fact_name values import_traces understands as-is. Its own default,
# "traces", is Okahu's name for the same thing import_traces calls "trace".
_PASSTHROUGH_FACT_NAMES = ("trace", "session")


def factid_import_kwargs(fact_id: FactID) -> dict:
    """Translate a FactID into MonocleValidator.import_traces keyword arguments.

    Any fact name that is not one import_traces recognizes is taken to be a
    custom scope name, matching how import_traces already treats the scope
    branch (it assigns the caller's scope_name straight to okahu_fact_name).

    ``workflow_name`` and ``load_spans`` are not derived here -- neither is a
    property of the fact, and the two call sites source them differently.

    Args:
        fact_id: The test case's FactID input.

    Returns:
        Keyword arguments for import_traces.

    Raises:
        ValueError: If the FactID carries no fact_id.
    """
    if not fact_id.fact_id:
        raise ValueError("fact_id is required to load a test case's trace")

    kwargs = {"trace_source": fact_id.source, "id": fact_id.fact_id}
    name = fact_id.fact_name
    if name in _PASSTHROUGH_FACT_NAMES:
        kwargs["fact_name"] = name
    elif name == "traces":
        kwargs["fact_name"] = "trace"
    else:
        kwargs["fact_name"] = "scope"
        kwargs["scope_name"] = name
    return kwargs


def _input_text(value: Any) -> Optional[str]:
    """A span's recorded input as one string, or None when it carries no prompt.

    ``data.input`` is a list far more often than a string -- usually the JSON
    message strings an agent was invoked with, sometimes dicts. A list becomes
    its elements joined by newlines, each stringified if it is not already text,
    with empty elements dropped: that reads as a prompt, where str() on the list
    would yield a Python literal (brackets, single quotes) and json.dumps would
    double-escape the already-JSON entries.

    Returns None for anything that yields no text, so an empty list is treated
    as "no input" rather than as the empty string.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, (list, tuple)):
        parts = [part if isinstance(part, str) else str(part)
                 for part in value if part]
        return "\n".join(parts) or None
    return str(value) or None


def turn_inputs_from_spans(spans: Sequence[ReadableSpan]) -> tuple:
    """The inputs a recorded run was driven with, in call order.

    Reads the turn spans, falling back to the first agent invocation when the
    spans hold none -- a partial trace, or a framework that emits no turn span.
    This is the same reading FluentTestCase.from_spans does for its own input.

    Args:
        spans: Spans of the recorded run, in any order.

    Returns:
        One entry per turn.

    Raises:
        ValueError: If the spans carry no input at all.
    """
    ordered = sorted(spans or [], key=lambda span: getattr(span, "start_time", 0) or 0)

    inputs = tuple(text for text in
                   (_input_text(get_input_from_span(span)) for span in ordered
                    if (span.attributes or {}).get("span.type") in TURN_SPAN_TYPES)
                   if text)
    if inputs:
        return inputs

    for span in ordered:
        if (span.attributes or {}).get("span.type") == SpanType.AGENTIC_INVOCATION.value:
            text = _input_text(get_input_from_span(span))
            if text:
                return (text,)

    raise ValueError(
        "the loaded trace has no turn or agent invocation input to replay")
