"""Pagination for the Okahu fact and trace enumerators.

The API pages with page_size out and next_page_token back. Monocle previously
sent neither, so every enumeration stopped at the server's DEFAULT_PAGE_SIZE of
100 with no log line, no tally and no exception -- backlog issue #242.
"""
import pytest

from monocle_test_tools.okahu_span_loader import OkahuSpanLoader


@pytest.fixture(autouse=True)
def _okahu_env(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "test-key")
    monkeypatch.setenv("OKAHU_API_ENDPOINT", "https://api.example")
    monkeypatch.delenv("OKAHU_API_TIMEOUT", raising=False)


class TestResolvePageSize:
    """Bounds are checked here so an out-of-range value never reaches the wire.

    The server answers a bad page_size with HTTP 400 rather than clamping, and
    that 400 surfaces mid-collection without naming the parameter that caused it.
    """

    def test_default_is_200(self):
        assert OkahuSpanLoader._resolve_page_size(None) == 200

    def test_an_explicit_size_is_honoured(self):
        assert OkahuSpanLoader._resolve_page_size(50) == 50

    def test_the_server_maximum_is_allowed(self):
        assert OkahuSpanLoader._resolve_page_size(1000) == 1000

    @pytest.mark.parametrize("bad", [0, -1, 1001])
    def test_out_of_range_raises_naming_the_bound(self, bad):
        with pytest.raises(ValueError, match="between 1 and 1000"):
            OkahuSpanLoader._resolve_page_size(bad)

    def test_a_string_raises(self):
        with pytest.raises(ValueError, match="must be an int"):
            OkahuSpanLoader._resolve_page_size("200")

    def test_a_bool_raises(self):
        """isinstance(True, int) is True in Python, so page_size=True would
        otherwise resolve to a page size of 1 rather than being rejected."""
        with pytest.raises(ValueError, match="must be an int"):
            OkahuSpanLoader._resolve_page_size(True)


def _envelope(ids, fact_count=None, next_page_token=None, key="traces"):
    """A response envelope shaped like the real API's."""
    body = {key: [{"trace_id": i} for i in ids]}
    if fact_count is not None:
        body["fact_count"] = fact_count
    if next_page_token is not None:
        body["next_page_token"] = next_page_token
    return body


def _extract_traces(envelope):
    """Pull trace ids out of an envelope, list-container style."""
    if not isinstance(envelope, dict):
        return []
    return [item["trace_id"] for item in envelope.get("traces", [])]


class TestIterPages:

    def test_the_token_is_sent_as_next_page_token(self):
        """The GET routes read 'next_page_token'; only the POST /evals/report
        pager spells it 'page_token'. Sending the wrong name is not an error --
        it is ignored and page 1 is re-served, which is indistinguishable from
        the bug being fixed. A mock that ignored param names could not catch it,
        so this one serves page 2 ONLY for the correct name.
        """
        calls = []

        def fetch(params):
            calls.append(dict(params))
            if params.get("next_page_token") is None:
                return _envelope(["a", "b"], fact_count=4, next_page_token="tok-2")
            assert params["next_page_token"] == "tok-2"
            return _envelope(["c", "d"], fact_count=4)

        collected = OkahuSpanLoader._collect_paged(
            fetch, {"start_time": "x"}, 200, _extract_traces, "traces in 'wf'")

        assert collected == ["a", "b", "c", "d"]
        assert "page_token" not in calls[1], "the POST pager's name must not be used"

    def test_page_size_is_sent_on_every_request(self):
        calls = []

        def fetch(params):
            calls.append(dict(params))
            if len(calls) == 1:
                return _envelope(["a"], fact_count=2, next_page_token="tok-2")
            return _envelope(["b"], fact_count=2)

        OkahuSpanLoader._collect_paged(fetch, {}, 50, _extract_traces, "ctx")

        assert [c["page_size"] for c in calls] == [50, 50]

    def test_a_single_page_issues_exactly_one_request(self):
        calls = []

        def fetch(params):
            calls.append(dict(params))
            return _envelope(["a", "b"], fact_count=2)

        collected = OkahuSpanLoader._collect_paged(
            fetch, {}, 200, _extract_traces, "ctx")

        assert collected == ["a", "b"]
        assert len(calls) == 1
        assert "next_page_token" not in calls[0]

    def test_three_pages_concatenate_in_server_order(self):
        pages = [
            _envelope(["a", "b"], fact_count=5, next_page_token="t2"),
            _envelope(["c", "d"], fact_count=5, next_page_token="t3"),
            _envelope(["e"], fact_count=5),
        ]

        def fetch(params):
            return pages.pop(0)

        assert OkahuSpanLoader._collect_paged(
            fetch, {}, 2, _extract_traces, "ctx") == ["a", "b", "c", "d", "e"]

    def test_an_empty_payload_gives_an_empty_list(self):
        collected = OkahuSpanLoader._collect_paged(
            lambda params: _envelope([], fact_count=0), {}, 200,
            _extract_traces, "ctx")

        assert collected == []

    def test_a_bare_list_envelope_does_not_crash(self):
        """Some mocks -- and the API's older shapes -- return a bare list rather
        than a dict. Reading the token off it must not raise AttributeError."""
        collected = OkahuSpanLoader._collect_paged(
            lambda params: [{"trace_id": "a"}], {}, 200,
            lambda env: [i["trace_id"] for i in env] if isinstance(env, list) else [],
            "ctx")

        assert collected == ["a"]

    def test_a_repeated_token_stops_the_walk(self, caplog):
        """A server echoing a token would spin the loop forever. Discovery runs
        at pytest collection time, so a hang is worse than a wrong answer."""
        calls = []

        def fetch(params):
            calls.append(dict(params))
            return _envelope(["a"], fact_count=99, next_page_token="same-token")

        collected = OkahuSpanLoader._collect_paged(
            fetch, {}, 200, _extract_traces, "traces in 'wf'")

        assert len(calls) == 2, "page 1, one follow-up, then stop"
        assert collected == ["a", "a"]
        assert "repeated a page token" in caplog.text


class TestReconciliation:

    def test_a_short_result_warns_naming_both_numbers(self, caplog):
        """The defect being fixed was silent. Never return short without saying so."""
        collected = OkahuSpanLoader._collect_paged(
            lambda params: _envelope(["a", "b"], fact_count=309), {}, 200,
            _extract_traces, "traces in 'wf'")

        assert collected == ["a", "b"]
        assert "2 of 309" in caplog.text
        assert "traces in 'wf'" in caplog.text

    def test_a_complete_result_is_silent(self, caplog):
        OkahuSpanLoader._collect_paged(
            lambda params: _envelope(["a", "b"], fact_count=2), {}, 200,
            _extract_traces, "ctx")

        assert "incomplete" not in caplog.text

    def test_no_fact_count_means_no_warning(self, caplog):
        """An envelope without fact_count advertises nothing to reconcile against."""
        OkahuSpanLoader._collect_paged(
            lambda params: _envelope(["a"]), {}, 200, _extract_traces, "ctx")

        assert "incomplete" not in caplog.text


class TestGetFactIdsPaginates:
    """/facts/<name>/ids keys fact_ids to a DICT, not a list -- the one shape
    difference from /traces. The protocol around it is identical."""

    @pytest.fixture(name="do_get")
    def do_get_fixture(self, monkeypatch):
        state = {"calls": [], "pages": []}

        def fake_do_get(url, headers, params=None, timeout=None, context_msg=""):
            state["calls"].append(dict(params or {}))
            return state["pages"].pop(0)

        monkeypatch.setattr(OkahuSpanLoader, "_do_get", staticmethod(fake_do_get))
        return state

    def test_it_walks_every_page(self, do_get):
        do_get["pages"] = [
            {"fact_ids": {"e-1": {}, "e-2": {}}, "fact_count": 3,
             "next_page_token": "tok-2"},
            {"fact_ids": {"e-3": {}}, "fact_count": 3},
        ]

        assert OkahuSpanLoader.get_fact_ids("wf", "agent_requests") == [
            "e-1", "e-2", "e-3"]
        assert do_get["calls"][1]["next_page_token"] == "tok-2"

    def test_it_sends_the_default_page_size(self, do_get):
        do_get["pages"] = [{"fact_ids": {}, "fact_count": 0}]

        OkahuSpanLoader.get_fact_ids("wf", "agent_requests")

        assert do_get["calls"][0]["page_size"] == 200

    def test_an_explicit_page_size_is_sent(self, do_get):
        do_get["pages"] = [{"fact_ids": {}, "fact_count": 0}]

        OkahuSpanLoader.get_fact_ids("wf", "agent_requests", page_size=25)

        assert do_get["calls"][0]["page_size"] == 25

    def test_a_bad_page_size_raises_before_any_request(self, do_get):
        do_get["pages"] = [{"fact_ids": {}, "fact_count": 0}]

        with pytest.raises(ValueError, match="between 1 and 1000"):
            OkahuSpanLoader.get_fact_ids("wf", "agent_requests", page_size=5000)

        assert do_get["calls"] == [], "no request may be issued"

    def test_an_unexpected_fact_ids_type_still_raises(self, do_get):
        do_get["pages"] = [{"fact_ids": "nonsense"}]

        with pytest.raises(ConnectionError, match="unexpected 'fact_ids'"):
            OkahuSpanLoader.get_fact_ids("wf", "agent_requests")


class TestGetTraceIdsPaginates:
    """/traces keys a LIST of trace objects. Same protocol as /facts/<n>/ids."""

    @pytest.fixture(name="do_get")
    def do_get_fixture(self, monkeypatch):
        state = {"calls": [], "pages": []}

        def fake_do_get(url, headers, params=None, timeout=None, context_msg=""):
            state["calls"].append(dict(params or {}))
            return state["pages"].pop(0)

        monkeypatch.setattr(OkahuSpanLoader, "_do_get", staticmethod(fake_do_get))
        return state

    def test_it_walks_every_page(self, do_get):
        do_get["pages"] = [
            _envelope(["t1", "t2"], fact_count=3, next_page_token="tok-2"),
            _envelope(["t3"], fact_count=3),
        ]

        assert OkahuSpanLoader.get_trace_ids("wf") == ["t1", "t2", "t3"]
        assert do_get["calls"][1]["next_page_token"] == "tok-2"

    def test_the_window_and_filter_survive_every_page(self, do_get):
        do_get["pages"] = [
            _envelope(["t1"], fact_count=2, next_page_token="tok-2"),
            _envelope(["t2"], fact_count=2),
        ]

        OkahuSpanLoader.get_trace_ids("wf", "agent_sessions", "sess_1",
                                      start_time="a", end_time="b",
                                      eval_filter="frustration")

        for call in do_get["calls"]:
            assert call["duration_fact"] == "agent_sessions"
            assert call["fact_ids"] == "sess_1"
            assert call["start_time"] == "a"
            assert call["eval"] == "frustration"

    def test_a_short_walk_warns(self, do_get, caplog):
        do_get["pages"] = [_envelope(["t1"], fact_count=309)]

        assert OkahuSpanLoader.get_trace_ids("wf") == ["t1"]
        assert "1 of 309" in caplog.text

    def test_a_bad_page_size_raises_before_any_request(self, do_get):
        do_get["pages"] = [_envelope([], fact_count=0)]

        with pytest.raises(ValueError, match="between 1 and 1000"):
            OkahuSpanLoader.get_trace_ids("wf", page_size=0)

        assert do_get["calls"] == []

    def test_half_a_fact_filter_still_raises_before_any_request(self, do_get):
        """The existing guard must run before the pager touches the network."""
        do_get["pages"] = [_envelope([], fact_count=0)]

        with pytest.raises(ValueError, match="fact_name and fact_id"):
            OkahuSpanLoader.get_trace_ids("wf", fact_name="agent_sessions")

        assert do_get["calls"] == []


class TestPageSizeThreading:
    """One knob. The eval report and the enumerators must not disagree about
    page depth, so setup_test_cases' page_size reaches all three."""

    @pytest.fixture(name="seen")
    def seen_fixture(self, monkeypatch):
        seen = {"trace": [], "fact": [], "report": []}

        def fake_trace_ids(workflow_name, fact_name=None, fact_id=None, **kwargs):
            seen["trace"].append(kwargs)
            return ["abc123"]

        def fake_fact_ids(workflow_name, fact_name, **kwargs):
            seen["fact"].append(kwargs)
            return ["e-1"]

        def fake_spans(*args, **kwargs):
            return []

        monkeypatch.setattr(OkahuSpanLoader, "get_trace_ids",
                            staticmethod(fake_trace_ids))
        monkeypatch.setattr(OkahuSpanLoader, "get_fact_ids",
                            staticmethod(fake_fact_ids))
        monkeypatch.setattr(OkahuSpanLoader, "get_spans", staticmethod(fake_spans))

        from monocle_test_tools.evals.okahu_eval import OkahuEval

        def fake_report(**kwargs):
            seen["report"].append(kwargs)
            return {}

        monkeypatch.setattr(OkahuEval, "_eval_report_by_fact",
                            staticmethod(fake_report))
        return seen

    def test_the_default_reaches_the_trace_enumerator(self, seen):
        OkahuSpanLoader.setup_test_cases(
            workflow_name="wf", start_time="a", end_time="b")

        assert seen["trace"][0]["page_size"] == 200

    def test_an_explicit_size_reaches_the_trace_enumerator(self, seen):
        OkahuSpanLoader.setup_test_cases(
            workflow_name="wf", start_time="a", end_time="b", page_size=25)

        assert seen["trace"][0]["page_size"] == 25

    def test_it_reaches_the_fact_enumerator(self, seen):
        OkahuSpanLoader.setup_test_cases(
            workflow_name="wf", start_time="a", end_time="b",
            fact_name="agentic_turns", page_size=25)

        assert seen["fact"][0]["page_size"] == 25

    def test_it_reaches_the_eval_report_too(self, seen):
        OkahuSpanLoader.setup_test_cases(
            workflow_name="wf", start_time="a", end_time="b",
            check_eval="sentiment", page_size=25)

        assert seen["report"][0]["page_size"] == 25

    def test_it_reaches_the_per_fact_trace_lookup(self, seen):
        """_fact_spans looks up one fact's traces; a fact spanning more than a
        page would otherwise load a partial span set."""
        OkahuSpanLoader.setup_test_cases(
            workflow_name="wf", start_time="a", end_time="b",
            fact_name="agentic_turns", page_size=25)

        assert seen["trace"][0]["page_size"] == 25


class TestResolveMaxFacts:
    """The ceiling on how many facts one discovery run may yield.

    Mirrors okahu_filtered_eval.py:319 -- same parameter name, same env var,
    same default -- so a deployment already setting OKAHU_MAX_FACTS gets the
    same bound from discovery.
    """

    def test_default_is_1000(self, monkeypatch):
        monkeypatch.delenv("OKAHU_MAX_FACTS", raising=False)

        assert OkahuSpanLoader._resolve_max_facts(None) == 1000

    def test_env_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv("OKAHU_MAX_FACTS", "50")

        assert OkahuSpanLoader._resolve_max_facts(None) == 50

    def test_an_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("OKAHU_MAX_FACTS", "50")

        assert OkahuSpanLoader._resolve_max_facts(25) == 25

    def test_an_argument_equal_to_the_default_is_still_honoured(self, monkeypatch):
        """A numeric default would make this indistinguishable from taking the
        default, which is why the signature defaults to None."""
        monkeypatch.setenv("OKAHU_MAX_FACTS", "50")

        assert OkahuSpanLoader._resolve_max_facts(1000) == 1000

    @pytest.mark.parametrize("bad_env", ["", "   ", "abc", "0", "-1", "1.5"])
    def test_an_unusable_env_value_is_logged_and_ignored(self, monkeypatch, caplog,
                                                         bad_env):
        """A misconfigured variable must not stop discovery -- same tolerance as
        OKAHU_API_TIMEOUT, and deliberately unlike okahu_filtered_eval's bare
        int() which would raise."""
        monkeypatch.setenv("OKAHU_MAX_FACTS", bad_env)

        assert OkahuSpanLoader._resolve_max_facts(None) == 1000
        if bad_env.strip():
            assert "OKAHU_MAX_FACTS" in caplog.text

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_a_non_positive_argument_raises(self, bad):
        with pytest.raises(ValueError, match="at least 1"):
            OkahuSpanLoader._resolve_max_facts(bad)

    def test_a_string_argument_raises(self):
        with pytest.raises(ValueError, match="must be an int"):
            OkahuSpanLoader._resolve_max_facts("1000")

    def test_a_bool_argument_raises(self):
        """isinstance(True, int) is True in Python, so max_facts=True would
        otherwise resolve to a ceiling of 1."""
        with pytest.raises(ValueError, match="must be an int"):
            OkahuSpanLoader._resolve_max_facts(True)


class TestMaxFactsCeiling:
    """A window yielding more facts than the ceiling is refused, not truncated.

    Truncating would hand back a silent subset -- the exact defect the
    pagination fix removes. The guard fails loudly and says how to proceed.
    """

    @pytest.fixture(name="seen")
    def seen_fixture(self, monkeypatch):
        """Stub the enumerators so a test can choose how many facts exist."""
        state = {"trace_kwargs": [], "fact_kwargs": [], "ids": ["t1", "t2"]}

        def fake_trace_ids(workflow_name, fact_name=None, fact_id=None, **kwargs):
            state["trace_kwargs"].append(kwargs)
            return list(state["ids"])

        def fake_fact_ids(workflow_name, fact_name, **kwargs):
            state["fact_kwargs"].append(kwargs)
            return list(state["ids"])

        monkeypatch.setattr(OkahuSpanLoader, "get_trace_ids",
                            staticmethod(fake_trace_ids))
        monkeypatch.setattr(OkahuSpanLoader, "get_fact_ids",
                            staticmethod(fake_fact_ids))
        monkeypatch.setattr(OkahuSpanLoader, "get_spans",
                            staticmethod(lambda *a, **k: []))
        return state

    def test_under_the_ceiling_passes(self, seen, monkeypatch):
        monkeypatch.delenv("OKAHU_MAX_FACTS", raising=False)
        seen["ids"] = [f"t{i}" for i in range(999)]

        cases = OkahuSpanLoader.setup_test_cases(
            workflow_name="wf", start_time="a", end_time="b")

        assert len(cases) == 999

    def test_exactly_at_the_ceiling_passes(self, seen, monkeypatch):
        """> ceiling, not >= : 1000 is allowed, matching okahu_filtered_eval."""
        monkeypatch.delenv("OKAHU_MAX_FACTS", raising=False)
        seen["ids"] = [f"t{i}" for i in range(1000)]

        cases = OkahuSpanLoader.setup_test_cases(
            workflow_name="wf", start_time="a", end_time="b")

        assert len(cases) == 1000

    def test_one_over_the_ceiling_raises(self, seen, monkeypatch):
        monkeypatch.delenv("OKAHU_MAX_FACTS", raising=False)
        seen["ids"] = [f"t{i}" for i in range(1001)]

        with pytest.raises(AssertionError, match="exceeding max_facts=1000"):
            OkahuSpanLoader.setup_test_cases(
                workflow_name="wf", start_time="a", end_time="b")

    def test_the_message_names_the_count_the_ceiling_and_the_env_var(self, seen):
        seen["ids"] = [f"t{i}" for i in range(3368)]

        with pytest.raises(AssertionError) as exc:
            OkahuSpanLoader.setup_test_cases(
                workflow_name="wf", start_time="a", end_time="b", max_facts=1000)

        message = str(exc.value)
        assert "3368" in message
        assert "max_facts=1000" in message
        assert "OKAHU_MAX_FACTS" in message
        assert "narrow the time window" in message
        assert "wf" in message

    def test_an_explicit_argument_is_honoured(self, seen):
        seen["ids"] = [f"t{i}" for i in range(11)]

        with pytest.raises(AssertionError, match="exceeding max_facts=10"):
            OkahuSpanLoader.setup_test_cases(
                workflow_name="wf", start_time="a", end_time="b", max_facts=10)

    def test_the_env_var_is_honoured(self, seen, monkeypatch):
        monkeypatch.setenv("OKAHU_MAX_FACTS", "5")
        seen["ids"] = [f"t{i}" for i in range(6)]

        with pytest.raises(AssertionError, match="exceeding max_facts=5"):
            OkahuSpanLoader.setup_test_cases(
                workflow_name="wf", start_time="a", end_time="b")

    def test_it_applies_to_non_trace_fact_levels_too(self, seen, monkeypatch):
        monkeypatch.delenv("OKAHU_MAX_FACTS", raising=False)
        seen["ids"] = [f"e-{i}" for i in range(1001)]

        with pytest.raises(AssertionError, match="exceeding max_facts=1000"):
            OkahuSpanLoader.setup_test_cases(
                workflow_name="wf", start_time="a", end_time="b",
                fact_name="agentic_turns")

    def test_a_bad_max_facts_raises_before_any_enumeration(self, seen):
        """Resolved before the enumerators run, so a bad argument costs no request."""
        with pytest.raises(ValueError, match="at least 1"):
            OkahuSpanLoader.setup_test_cases(
                workflow_name="wf", start_time="a", end_time="b", max_facts=0)

        assert seen["trace_kwargs"] == [], "no enumeration may be issued"

    def test_the_enumerators_are_not_given_a_ceiling(self, seen):
        """The whole design: the ceiling never enters the shared span-loading
        plumbing, so no other caller of get_trace_ids/get_fact_ids can inherit
        it. See load_by_scope, import_traces, from_okahu_scope."""
        OkahuSpanLoader.setup_test_cases(
            workflow_name="wf", start_time="a", end_time="b", max_facts=10)

        assert "max_facts" not in seen["trace_kwargs"][0]


class TestPlumbingHasNoCeiling:
    """Structural guard: if someone later threads max_facts into the shared
    loaders, these fail. That coupling is what made the first attempt at this
    feature silently cap load_by_scope and _fact_spans."""

    @pytest.mark.parametrize("name", ["get_trace_ids", "get_fact_ids",
                                      "_collect_paged", "_iter_pages"])
    def test_the_shared_loaders_take_no_max_facts(self, name):
        import inspect

        params = inspect.signature(getattr(OkahuSpanLoader, name)).parameters
        assert "max_facts" not in params, (
            f"{name} must stay free of the ceiling -- it is shared with "
            f"load_by_scope, import_traces and from_okahu_scope")

    def test_feature_b_loads_more_traces_than_the_ceiling(self, monkeypatch):
        """load_by_scope fetches one scope's traces. It must not be bounded by a
        discovery ceiling, no matter how many traces that scope holds."""
        monkeypatch.delenv("OKAHU_MAX_FACTS", raising=False)
        trace_ids = [f"t{i}" for i in range(1500)]

        def fake_do_get(url, headers, params=None, timeout=None, context_msg=""):
            return _envelope(trace_ids, fact_count=1500)

        monkeypatch.setattr(OkahuSpanLoader, "_do_get", staticmethod(fake_do_get))
        monkeypatch.setattr(OkahuSpanLoader, "get_spans",
                            staticmethod(lambda *a, **k: ["span"]))

        spans = OkahuSpanLoader.load_by_scope(
            workflow_name="wf", scope_name="test_id", scope_id="run-42")

        assert len(spans) == 1500
