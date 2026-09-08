"""Live proof that the enumerators walk past the server's first page.

Against stage, not prod. The assertions are self-consistency checks rather than
literals: the workflow keeps ingesting, so "== 810" would rot within days, while
"collected everything the server says exists" stays true forever.

Credentials and the stage endpoint come from tests/integration/__init__.py.

At the time of writing this window holds 810 traces and 809 agent_requests, both
far past the server's default page of 100. Before the fix both enumerators
returned exactly 100 and said nothing about the rest.

Backlog issue #242.
"""
import os

import pytest

from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

# A workflow, not an app: Monocle addresses Okahu by workflow name throughout --
# the span loader, the eval report and the pytest plugin all do. The window is
# deliberately wide so both fact levels span several pages.
WORKFLOW = "karmehr-okahu-monocle"
START, END = "2025-01-01T00:00:00.000Z", "2026-09-03T23:59:59.000Z"
SERVER_DEFAULT_PAGE = 100

# Running this against prod would walk an unrelated tenant, so the endpoint is
# checked rather than assumed -- _get_api_base falls back to the prod base URL
# when nothing is configured.
_ON_STAGE = "stage" in OkahuSpanLoader._get_api_base()

pytestmark = [
    pytest.mark.skipif(not os.environ.get("OKAHU_API_KEY"),
                       reason="needs OKAHU_API_KEY"),
    pytest.mark.skipif(not _ON_STAGE,
                       reason=f"needs the stage endpoint; got "
                              f"{OkahuSpanLoader._get_api_base()}"),
]


def _advertised(path_suffix, params):
    """fact_count straight off page 1, before any paging."""
    envelope = OkahuSpanLoader._get_resource(
        OkahuSpanLoader._get_api_base(), path_suffix,
        OkahuSpanLoader._get_headers(),
        params={"start_time": START, "end_time": END, **params},
        context_msg="probe")
    return envelope.get("fact_count")


def test_trace_enumeration_collects_every_page():
    advertised = _advertised(f"{WORKFLOW}/traces", {})
    assert advertised > SERVER_DEFAULT_PAGE, (
        "this window must span more than one page or the test proves nothing")

    ids = OkahuSpanLoader.get_trace_ids(WORKFLOW, start_time=START, end_time=END)

    assert len(ids) == advertised, "collected everything the server advertises"
    assert len(ids) > SERVER_DEFAULT_PAGE, "the 100-row truncation is gone"
    assert len(set(ids)) == len(ids), (
        "duplicates mean the token re-served page 1 instead of advancing")


def test_fact_enumeration_collects_every_page():
    fact = "agent_requests"
    advertised = _advertised(
        f"{WORKFLOW}/facts/{fact}/ids",
        {"duration_fact": fact, "breakdown_filter": fact})
    assert advertised > SERVER_DEFAULT_PAGE

    ids = OkahuSpanLoader.get_fact_ids(WORKFLOW, fact,
                                       start_time=START, end_time=END)

    assert len(ids) == advertised
    assert len(ids) > SERVER_DEFAULT_PAGE
    assert len(set(ids)) == len(ids)


def test_an_explicit_page_size_reaches_the_same_total():
    """Page depth must not change the answer -- only how many round trips it
    takes. A smaller page forces more token follow-ups over the same window."""
    wide = OkahuSpanLoader.get_trace_ids(WORKFLOW, start_time=START, end_time=END,
                                         page_size=1000)
    narrow = OkahuSpanLoader.get_trace_ids(WORKFLOW, start_time=START, end_time=END,
                                           page_size=25)

    assert sorted(wide) == sorted(narrow)


def test_the_default_ceiling_refuses_an_oversized_window(monkeypatch):
    """This window holds ~3368 inferences, comfortably past the default 1000,
    so the ceiling engages against real data rather than a mock."""
    monkeypatch.delenv("OKAHU_MAX_FACTS", raising=False)

    with pytest.raises(AssertionError, match="exceeding max_facts=1000"):
        OkahuSpanLoader.setup_test_cases(
            workflow_name=WORKFLOW, start_time=START, end_time=END,
            fact_name="inferences")


def test_raising_the_ceiling_lets_the_same_window_through(monkeypatch):
    """Enumeration alone must still work at that size -- only span loading is
    expensive, and setup_test_cases is not reached here."""
    monkeypatch.delenv("OKAHU_MAX_FACTS", raising=False)

    ids = OkahuSpanLoader.get_fact_ids(WORKFLOW, "inferences",
                                       start_time=START, end_time=END)

    assert len(ids) > 1000, "the window must exceed the default to prove anything"
    assert len(set(ids)) == len(ids)


def test_the_env_var_overrides_the_default(monkeypatch):
    monkeypatch.setenv("OKAHU_MAX_FACTS", "5")

    with pytest.raises(AssertionError, match="exceeding max_facts=5"):
        OkahuSpanLoader.setup_test_cases(
            workflow_name=WORKFLOW, start_time=START, end_time=END)


# Narrower than START: ~58 traces, comfortably under the default ceiling.
NARROW_START = "2026-06-01T00:00:00.000Z"


def test_a_window_under_the_ceiling_still_builds_every_case(monkeypatch):
    """The happy path, live.

    The other two ceiling tests both assert refusal, which would still pass if
    the guard rejected everything. This one proves a window under the limit is
    undisturbed: every advertised fact becomes a loaded case.
    """
    monkeypatch.delenv("OKAHU_MAX_FACTS", raising=False)

    advertised = _advertised(f"{WORKFLOW}/traces", {"start_time": NARROW_START})
    assert advertised < 1000, (
        "this window must sit under the ceiling or the test proves nothing")

    cases = OkahuSpanLoader.setup_test_cases(
        workflow_name=WORKFLOW, start_time=NARROW_START, end_time=END)

    assert len(cases) == advertised, "every fact under the ceiling becomes a case"
    assert all(c.load_error is None for c in cases), "no case may fail to load"
