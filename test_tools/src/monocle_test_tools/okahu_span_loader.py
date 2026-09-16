import logging
import os
from typing import Any, Dict, List, Optional, Union
import requests
from opentelemetry.sdk.trace import ReadableSpan
from monocle_test_tools.file_span_loader import JSONSpanLoader

logger = logging.getLogger(__name__)


class OkahuSpanLoader:
    """Utility class to load spans from Okahu trace service.

    Uses the Okahu REST API:
        - GET /api/v1/apps/<app_name>/traces?duration_fact=<fact>&fact_ids=<id>
          Get traces matching a fact (e.g. ``agent_sessions``).
        - GET /api/v1/apps/<app_name>/traces/<trace_id>/spans
          Get spans for a trace, optionally filtered by session.

    Base URL defaults to https://api.okahu.co and can be overridden
    with the OKAHU_API_ENDPOINT environment variable.
    """

    # Constants
    AGENT_SESSIONS_SCOPE = "agent_sessions"
    OKAHU_BASE_URL = "https://api.okahu.co"

    RESOURCE_NAMESPACES = ("apps", "workflows")

    # Okahu calls can be slow: a fact above trace level fans out over several
    # traces, and the eval report paginates. 120s is the floor that stopped
    # those timing out; OKAHU_API_TIMEOUT raises or lowers it per deployment.
    DEFAULT_API_TIMEOUT = 120

    # The server's own DEFAULT_PAGE_SIZE is 100 (common/api/util.py:34) and its
    # MAX_PAGE_SIZE is 1000, enforced with a 400 rather than a clamp. 200 halves
    # the round trips and matches what the eval-tune-kit clients already request.
    DEFAULT_PAGE_SIZE = 200
    MAX_PAGE_SIZE = 1000

    # The ceiling okahu_filtered_eval applies to a filtered run (evals/
    # okahu_filtered_eval.py:319), so both paths bound a window the same way.
    DEFAULT_MAX_FACTS = 1000

    @staticmethod
    def _get_api_base(endpoint: Optional[str] = None) -> str:
        """Return the Okahu API base URL (no trailing slash).

        ``or OKAHU_BASE_URL`` rather than a getenv default: the variable is
        set-but-empty under pytest (tests/integration/__init__.py setdefaults it
        to ""), and an empty base builds a hostless URL that fails with
        MissingSchema instead of falling back to prod.
        """
        return (endpoint or os.environ.get("OKAHU_API_ENDPOINT")
                or OkahuSpanLoader.OKAHU_BASE_URL).rstrip("/")

    @staticmethod
    def _resolve_timeout(timeout: Optional[int] = None) -> int:
        """Seconds to allow a request: explicit argument, else env, else default.

        Every caller defaults ``timeout`` to None rather than to a number, which
        is what makes the precedence expressible -- a numeric default would be
        indistinguishable from a caller asking for that number, and would
        silently outrank OKAHU_API_TIMEOUT.

        An unusable value in the environment (empty, non-numeric, or not a
        positive integer) is logged and ignored: a misconfigured variable should
        not stop span loading.
        """
        if timeout is not None:
            return timeout

        raw = (os.environ.get("OKAHU_API_TIMEOUT") or "").strip()
        if not raw:
            return OkahuSpanLoader.DEFAULT_API_TIMEOUT
        try:
            seconds = int(raw)
        except ValueError:
            seconds = 0
        if seconds <= 0:
            logger.warning(
                "OKAHU_API_TIMEOUT=%r is not a positive integer; using %ds",
                raw, OkahuSpanLoader.DEFAULT_API_TIMEOUT)
            return OkahuSpanLoader.DEFAULT_API_TIMEOUT
        return seconds

    @staticmethod
    def _resolve_page_size(page_size: Optional[int] = None) -> int:
        """Rows per page: an explicit value if given, else DEFAULT_PAGE_SIZE.

        parse_page_size on the server (common/api/util.py:558-570) answers an
        out-of-range value with HTTP 400 rather than clamping, and that 400
        surfaces mid-collection with no indication of which parameter caused it.
        Failing here names the bound instead.

        No environment override, unlike the timeout: a timeout is a deployment
        property, while page size is a tuning detail already exposed as a
        parameter on setup_test_cases.
        """
        if page_size is None:
            return OkahuSpanLoader.DEFAULT_PAGE_SIZE
        # bool before int: isinstance(True, int) is True, so page_size=True would
        # otherwise pass as a page size of 1.
        if isinstance(page_size, bool) or not isinstance(page_size, int):
            raise ValueError(
                f"page_size must be an int, got {type(page_size).__name__}")
        if not 1 <= page_size <= OkahuSpanLoader.MAX_PAGE_SIZE:
            raise ValueError(
                f"page_size must be between 1 and {OkahuSpanLoader.MAX_PAGE_SIZE} "
                f"(the Okahu server's MAX_PAGE_SIZE), got {page_size}")
        return page_size

    @staticmethod
    def _resolve_max_facts(max_facts: Optional[int] = None) -> int:
        """The most facts one discovery run may yield: argument, else env, else 1000.

        Mirrors the ceiling okahu_filtered_eval applies to a filtered run
        (evals/okahu_filtered_eval.py:319), so a deployment already setting
        OKAHU_MAX_FACTS gets the same bound here. Kept as a separate
        implementation rather than a shared import because okahu_span_loader and
        the evals modules form an import cycle -- see the local imports in
        setup_test_cases.

        An unusable OKAHU_MAX_FACTS -- empty, non-numeric, or not positive -- is
        logged and ignored rather than stopping discovery, matching how
        OKAHU_API_TIMEOUT resolves. Deliberately more tolerant than
        okahu_filtered_eval, whose bare int() would raise on a bad value.
        """
        if max_facts is not None:
            # bool before int: isinstance(True, int) is True, so max_facts=True
            # would otherwise pass as a ceiling of 1.
            if isinstance(max_facts, bool) or not isinstance(max_facts, int):
                raise ValueError(
                    f"max_facts must be an int, got {type(max_facts).__name__}")
            if max_facts < 1:
                raise ValueError(f"max_facts must be at least 1, got {max_facts}")
            return max_facts

        raw = (os.environ.get("OKAHU_MAX_FACTS") or "").strip()
        if not raw:
            return OkahuSpanLoader.DEFAULT_MAX_FACTS
        try:
            ceiling = int(raw)
        except ValueError:
            ceiling = 0
        if ceiling < 1:
            logger.warning(
                "OKAHU_MAX_FACTS=%r is not a positive integer; using %d",
                raw, OkahuSpanLoader.DEFAULT_MAX_FACTS)
            return OkahuSpanLoader.DEFAULT_MAX_FACTS
        return ceiling

    @staticmethod
    def _get_headers(api_key: Optional[str] = None) -> dict:
        """Return common request headers."""
        key = api_key or os.environ.get("OKAHU_API_KEY")
        if not key:
            raise ValueError("OKAHU_API_KEY is not configured. Set the environment variable or pass api_key.")
        return {
            "Content-Type": "application/json",
            "x-api-key": key
        }

    @staticmethod
    def _get_resource(base: str, path_suffix: str, headers: dict,
                      params: Optional[dict] = None, timeout: Optional[int] = None,
                      context_msg: str = "") -> Any:
        """GET an application-scoped resource, trying each known namespace.

        ``path_suffix`` is everything after the application name, e.g.
        ``"traces"`` or ``"traces/<id>/spans"``. Only a 404 moves on to the next
        namespace; any other error is the caller's to see.
        """
        last_error: Optional[requests.HTTPError] = None
        for namespace in OkahuSpanLoader.RESOURCE_NAMESPACES:
            url = f"{base}/api/v1/{namespace}/{path_suffix}"
            try:
                return OkahuSpanLoader._do_get(
                    url, headers, params=params, timeout=timeout, context_msg=context_msg
                )
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status != 404:
                    raise
                logger.debug("Okahu %r namespace returned 404 for %s", namespace, path_suffix)
                last_error = exc
        raise last_error

    @staticmethod
    def _do_get(url: str, headers: dict, params: Optional[dict] = None,
                timeout: Optional[int] = None, context_msg: str = "") -> Any:
        """Execute a GET request with standard error handling."""
        try:
            response = requests.get(url=url, headers=headers, params=params,
                                    timeout=OkahuSpanLoader._resolve_timeout(timeout))
            response.raise_for_status()
        except requests.Timeout as exc:
            raise ConnectionError(f"Okahu request timed out ({context_msg}): {exc}") from exc
        except requests.HTTPError as exc:
            raise
        except requests.RequestException as exc:
            raise ConnectionError(f"Failed to reach Okahu service ({context_msg}): {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise ConnectionError(
                f"Okahu returned invalid JSON ({context_msg}): {response.text}"
            ) from exc

    @staticmethod
    def _window_params(start_time: Optional[str], end_time: Optional[str]) -> dict:
        """The time-window query params, omitting whichever end was not given.

        Returned as a dict to merge so that a call with no window leaves `params`
        exactly as it was -- callers rely on an empty params becoming None.
        """
        window = {}
        if start_time is not None:
            window["start_time"] = start_time
        if end_time is not None:
            window["end_time"] = end_time
        return window

    @staticmethod
    def _eval_param(eval_filter: Optional[str]) -> dict:
        """The eval query param, omitted when no filter is given.

        Sent raw -- requests percent-encodes it -- so the value may be a bare
        eval name or the API's ``name:label;name:label`` form.
        """
        return {"eval": eval_filter} if eval_filter else {}

    @staticmethod
    def _unwrap_list(data: Any, wrapper_keys: tuple, context_msg: str = "") -> list:
        """Unwrap a list from a possible dict wrapper."""
        if isinstance(data, dict):
            for key in wrapper_keys:
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ConnectionError(
                f"Okahu response is a dict but no known list key found ({context_msg}). "
                f"Keys: {list(data.keys())}"
            )
        if isinstance(data, list):
            return data
        raise ConnectionError(
            f"Expected a list from Okahu ({context_msg}), got: {type(data).__name__}"
        )

    @staticmethod
    def _iter_pages(fetch, params: dict, page_size: int, context_msg: str = ""):
        """Yield each response envelope, following the page token to exhaustion.

        A GET/query-params sibling of okahu_eval._iter_eval_report_rows, which
        walks the same protocol over POST/body. Both endpoints signal the end by
        omitting next_page_token.

        ``fetch`` is a callable taking the params dict and returning the parsed
        envelope. It is supplied by the caller because get_trace_ids resolves its
        namespace through _get_resource while get_fact_ids has a fixed URL.

        The envelope is yielded whole rather than its items, so a caller wanting
        something other than ids -- get_spans, eventually -- can reuse this.
        """
        page_token, seen_tokens = None, set()
        while True:
            page_params = {**params, "page_size": page_size}
            if page_token:
                # These GET routes read the token as 'next_page_token'; only the
                # POST /evals/report pager spells it 'page_token'. The wrong name
                # is not an error -- it is ignored and page 1 is re-served.
                # Verified in okahu/routes/{traces,facts}.py, both of which do:
                #   req.query_params.get('next_page_token') or ...('prev_page_token')
                page_params["next_page_token"] = page_token
            envelope = fetch(page_params)
            yield envelope

            # An envelope is not always a dict: some responses are a bare list.
            page_token = (envelope.get("next_page_token")
                          if isinstance(envelope, dict) else None)
            if not page_token:
                return
            if page_token in seen_tokens:
                logger.warning(
                    "Okahu repeated a page token (%s); stopping the walk after "
                    "%d pages", context_msg, len(seen_tokens) + 1)
                return
            seen_tokens.add(page_token)

    @staticmethod
    def _collect_paged(fetch, params: dict, page_size: int, extract,
                       context_msg: str = "") -> list:
        """Every item across every page, with a warning if the total falls short.

        ``extract`` turns one envelope into its items, which is where the two
        endpoints differ: /traces keys a list, /facts/<n>/ids keys a dict.

        fact_count is read from every page, so a mid-walk revision by the server
        takes effect -- and it costs nothing, the envelope is already open.
        """
        items, advertised = [], None
        for envelope in OkahuSpanLoader._iter_pages(
                fetch, params, page_size, context_msg):
            items.extend(extract(envelope))
            if isinstance(envelope, dict) and isinstance(
                    envelope.get("fact_count"), int):
                advertised = envelope["fact_count"]

        if advertised is not None and len(items) != advertised:
            # The defect this exists to kill was silent. Never return a short
            # list without saying so.
            logger.warning(
                "Okahu returned %d of %d advertised (%s); the result is incomplete",
                len(items), advertised, context_msg)
        return items

    # ------------------------------------------------------------------ #
    #  Public helpers                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def setup_test_cases(cls, *, workflow_name: str, start_time: str, end_time: str,
                         fact_name: str = "traces",
                         category: Union[str, list] = "llm",
                         eval_filter: Optional[str] = None,
                         check_eval: Optional[Union[bool, str]] = None,
                         compare_eval: Optional[str] = None,
                         page_size: int = DEFAULT_PAGE_SIZE,
                         max_facts: Optional[int] = None) -> list:
        """Build FluentTestCases from the traces recorded for a workflow.

        One test case per fact in the window. How the facts are found depends on
        the level:

        - ``traces`` (the default): a trace is its own fact, so
          ``OkahuSpanLoader.get_trace_ids`` enumerates them and each has one
          trace's spans.
        - anything above a trace (agent requests, sessions, ...):
          ``get_fact_ids`` enumerates them from the fact ids API, then each
          fact's traces are looked up and *all* their spans concatenated -- the
          combined set is what describes that fact.

        Either way the spans are read by ``FluentTestCase.from_spans``, which
        fills in the agents invoked, the tools they called and the tokens
        consumed. Finally, only when ``check_eval`` is set,
        ``/v1/workflows/{workflow_name}/evals/report`` is asked about exactly those
        traces (``fact_ids``) and that eval, and the labels it returns become each
        case's expected results.

        The case's ``input`` stays the FactID rather than the prompt from_spans
        would derive: ``with_trace_source(testcase=...)`` needs a FactID, and
        ``run_agent(testcase=...)`` resolves one into the prompt itself.

        This is one request per trace plus one for the report, and above trace
        level another per fact to find its traces -- so a wide window is a lot of
        calls.

        Sending ``fact_ids`` takes the report endpoint OUT of discovery mode -- the
        absence of fact_ids is what selects discovery -- so it reports on traces
        already enumerated rather than re-discovering them.

        Without ``check_eval`` no report call is made at all and the cases
        carry no evals -- still fully described otherwise, and ready for
        ``run_agent(testcase=...)`` to replay. With one, a fact that has no
        labelled result for that eval is
        dropped: an empty ``evals`` list raises in ``check_eval``, so emitting the
        case would poison the suite it is meant to feed. The dropped count is
        logged rather than passed over in silence.

        Either way the result is directly parametrizable, being the shape
        ``with_trace_source(testcase=...)`` and ``check_eval(testcase=...)`` consume.

        A row's label is taken from ``authoritative.eval_result.label``, falling
        back to the newest entry in ``latest``. Rows with neither are skipped, and a
        fact left with no labelled eval yields no test case at all -- an empty
        ``evals`` list would raise in check_eval, so emitting one would only
        manufacture a broken case.

        Custom evals (``eval_id`` prefixed ``custom_evaluation__``) are returned
        like any other, by name. Okahu does not store their templates, so if such a
        name does not also resolve as a stored template, the case will fail when
        check_eval runs it rather than being filtered out here -- deliberate, so the
        report is reflected as-is instead of silently shrinking.

        Args:
            workflow_name: Okahu workflow / service name.
            start_time: Window start. Required -- discovery has no silent default.
            end_time: Window end. Required.
            fact_name: User-facing fact level, mapped to the Okahu name for the
                request. The returned FactIDs keep the user-facing name.
            category: Which eval runs to consider -- ``"llm"`` (the default),
                ``"manual"`` or ``"test"``, or a list of them. A bare string is
                wrapped, so this is always sent as a list.
            eval_filter: Optional ``eval`` filter narrowing which facts are
                considered at all -- passed to the fact/trace lookups as a query
                param. A bare eval name, or the API's ``name:label;name:label``
                form. Does not by itself cause any eval to be reported.
            check_eval: Which evals to report on, as a switch or a name.
                ``True`` reports every eval recorded for the fact level, a
                string reports only that one, and ``False``/omitted makes no
                report call at all. The labels become each case's expected
                results. ``"custom"`` is rejected: the report resolves a stored
                template by name and custom templates are not stored.
            compare_eval: Take the expected result from a *different* eval. The
                report is asked about this one and its labels are used, but each
                case still names ``check_eval`` -- so the case reads "run
                check_eval, expect what compare_eval recorded". That is the
                eval-tuning question: does a new template reproduce a golden
                one's labels? Requires ``check_eval`` to be a name, since there
                is otherwise nothing to attach the borrowed label to.
            page_size: Rows per page for the trace/fact enumeration and the eval
                report alike -- one knob, so they cannot disagree about page
                depth. Server maximum 1000.
            max_facts: The most facts this window may yield. Defaults to
                OKAHU_MAX_FACTS, then DEFAULT_MAX_FACTS (1000). A wider window
                raises, naming the count and the variable, rather than
                generating thousands of cases -- each fact costs a span request
                and, with check_eval, an eval. Bounds the fact enumeration only;
                the spans beneath one fact are not counted.

        Returns:
            One FluentTestCase per fact that has at least one labelled eval.

        Raises:
            ValueError: If check_eval is "custom", or fact_name is not
                recognized.
            AssertionError: If the report service cannot be reached or errors.
        """
        # Local imports: monocle_test_tools.schema is still partially initialized
        # when this module is first imported (schema -> ... -> evals -> okahu_eval),
        # so importing FactID at module scope raises. Verified, not precautionary.
        from monocle_test_tools.evals.okahu_filtered_eval import normalize_fact_id
        from monocle_test_tools.schema import FactID
        from monocle_test_tools.testcase import FluentTestCase

        if "custom" in (check_eval, compare_eval):
            raise ValueError(
                "'custom' is not supported for check_eval or compare_eval; the "
                "report resolves a stored template by name and custom templates "
                "are not stored.")
        if compare_eval and not isinstance(check_eval, str):
            raise ValueError(
                "compare_eval needs check_eval to name a single eval: its label is "
                "borrowed as the expected result for check_eval, so there must be "
                f"one name to attach it to (got check_eval={check_eval!r}).")

        # Local import: okahu_eval imports this module, so importing it back at
        # module scope would be a cycle. The eval report stays with the
        # evaluator; only the span gathering belongs here.
        from monocle_test_tools.evals.okahu_eval import OkahuEval

        mapped_fact_name = OkahuEval._map_fact_name(fact_name)
        # A trace IS its own fact, so the trace list is the fact list. Any level
        # above a trace -- agent requests, sessions -- is enumerated by its own
        # ids API, and each of those facts spans one or more traces.
        # Resolved before enumeration so a bad argument fails without a request.
        ceiling = cls._resolve_max_facts(max_facts)

        if mapped_fact_name == "traces":
            fact_ids = [normalize_fact_id(tid) for tid in cls.get_trace_ids(
                workflow_name, start_time=start_time, end_time=end_time,
                eval_filter=eval_filter, page_size=page_size)]
        else:
            fact_ids = cls.get_fact_ids(
                workflow_name, mapped_fact_name,
                start_time=start_time, end_time=end_time,
                eval_filter=eval_filter, page_size=page_size)

        # Paginating the enumerators removed an accidental ceiling: they used to
        # stop at the server's first page of 100. Every fact here costs a span
        # request and, with check_eval, an eval, so an oversized window is
        # refused before any of that -- never truncated, which would hand back a
        # silent subset.
        #
        # Deliberately checked HERE rather than inside the enumerators: they are
        # shared with load_by_scope, import_traces and from_okahu_scope, which
        # load one named fact and have no runaway to guard against.
        if len(fact_ids) > ceiling:
            raise AssertionError(
                f"Okahu discovered {len(fact_ids)} '{fact_name}' facts in workflow "
                f"'{workflow_name}', exceeding max_facts={ceiling} (set "
                f"OKAHU_MAX_FACTS to raise the ceiling, or narrow the time window).")

        if not fact_ids:
            return []

        # One bulk call for every fact, so a failure here belongs to all of them
        # rather than to any one: record it on each and carry on, so the window
        # still yields a suite that reports the outage instead of no suite.
        evals_by_fact, eval_error = {}, None
        if check_eval:
            # A string names one eval; True asks for every eval the fact level
            # supports, which the report expresses by omitting eval_names.
            try:
                evals_by_fact = OkahuEval._eval_report_by_fact(
                    workflow_name=workflow_name, fact_ids=fact_ids,
                    fact_name=mapped_fact_name, start_time=start_time,
                    end_time=end_time, category=category,
                    eval_name=compare_eval or (
                        check_eval if isinstance(check_eval, str) else None),
                    name_as=check_eval if compare_eval else None,
                    page_size=page_size)
            except cls._LOAD_ERRORS as exc:
                eval_error = f"could not load evals for the window: {exc}"
                logger.warning("setup_test_cases: %s", eval_error)

        test_cases, dropped, failed = [], 0, 0
        for fact_id in fact_ids:
            fact = FactID(fact_id=fact_id, fact_name=fact_name, source="okahu")
            if eval_error:
                test_cases.append(FluentTestCase(
                    name=fact_id, input=fact, load_error=eval_error))
                failed += 1
                continue

            evals = evals_by_fact.get(fact_id, [])
            if check_eval and not evals:
                # Asked for a specific eval and this trace has no labelled result
                # for it. An empty evals list raises in check_eval, so emitting the
                # case would poison the suite it is meant to feed. This is a fact
                # that loaded fine and simply has no such eval -- not a failure.
                dropped += 1
                continue

            # from_spans reads the agents, tools and token count off the spans;
            # name and input are supplied here. input stays the FactID rather than
            # the recorded prompt from_spans would derive: with_trace_source
            # needs a FactID and run_agent resolves one into the prompt anyway.
            #
            # A fact whose spans will not load is emitted carrying the reason
            # instead of aborting: this runs at collection time, so raising would
            # cost every other fact in the window its test.
            try:
                spans = cls._fact_spans(
                    workflow_name, fact_id, mapped_fact_name=mapped_fact_name,
                    start_time=start_time, end_time=end_time,
                    eval_filter=eval_filter, page_size=page_size)
            except cls._LOAD_ERRORS as exc:
                test_cases.append(FluentTestCase(
                    name=fact_id, input=fact,
                    load_error=f"could not load spans for fact '{fact_id}': {exc}"))
                failed += 1
                continue

            test_cases.append(FluentTestCase.from_spans(
                spans, name=fact_id, input=fact, evals=evals))

        if dropped:
            logger.info("setup_test_cases: %d of %d facts had no labelled '%s' eval "
                        "and were dropped", dropped, len(fact_ids), check_eval)
        if failed:
            logger.warning("setup_test_cases: %d of %d facts could not be loaded; "
                           "each is reported as a failing test case",
                           failed, len(fact_ids))
        return test_cases

    # Errors that mean "this data would not load", as opposed to a bug. The
    # loaders raise ConnectionError for transport trouble and re-raise
    # requests.HTTPError for a non-404 status; the eval report raises
    # AssertionError. Anything else is left to propagate.
    _LOAD_ERRORS = (AssertionError, ConnectionError, requests.RequestException, ValueError)

    @classmethod
    def _fact_spans(cls, workflow_name, fact_id, *, mapped_fact_name,
                    start_time, end_time, eval_filter=None,
                    page_size=None) -> list:
        """Every span belonging to one fact.

        A trace-level fact is one trace, so its spans are one call. A higher
        level fact spans one or more traces, so its traces are looked up and
        their spans concatenated -- the combined set is what describes the fact.
        """
        if mapped_fact_name == "traces":
            return cls.get_spans(
                workflow_name, fact_id, start_time=start_time, end_time=end_time)

        spans = []
        for trace_id in cls.get_trace_ids(
                workflow_name, mapped_fact_name, fact_id,
                start_time=start_time, end_time=end_time,
                eval_filter=eval_filter, page_size=page_size):
            spans.extend(cls.get_spans(
                workflow_name, trace_id, start_time=start_time, end_time=end_time))
        return spans



    @staticmethod
    def get_fact_ids(
        workflow_name: str,
        fact_name: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        eval_filter: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> List[str]:
        """Fetch the ids of every fact of one level in a workflow.

        Uses:  GET /api/v1/workflows/<wf>/facts/<fact_name>/ids
               ?duration_fact=<fact_name>&breakdown_filter=<fact_name>

        This is the entry point for any fact level above a trace -- agent
        requests, sessions, conversations. A trace-level set comes from
        ``get_trace_ids`` instead.

        The response keys ``fact_ids`` to an object keyed by id, not to a list,
        so the ids are its keys and the order is the server's. Each value holds
        that fact's timing and status, and *sometimes* a ``traces`` array -- only
        for the first entry, in practice. That array is deliberately ignored:
        relying on it would make one fact behave differently from the rest, so
        every fact's traces are fetched uniformly with ``get_trace_ids``.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            fact_name: The Okahu fact level (e.g. ``agent_requests``), already
                mapped -- this goes straight into the URL path.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds. Defaults to
                OKAHU_API_TIMEOUT, then ``DEFAULT_API_TIMEOUT`` (120).
            start_time: Optional window start.
            end_time: Optional window end.
            eval_filter: Optional ``eval`` filter narrowing the result set to
                facts carrying it -- a bare eval name, or the API's
                ``name:label;name:label`` form.
            page_size: Rows per page. Defaults to DEFAULT_PAGE_SIZE (200);
                the server rejects anything above MAX_PAGE_SIZE (1000).

        Returns:
            The fact ids across every page, in the order the server returned them.
        """
        page_size = OkahuSpanLoader._resolve_page_size(page_size)
        base = OkahuSpanLoader._get_api_base(endpoint)
        headers = OkahuSpanLoader._get_headers(api_key)
        url = f"{base}/api/v1/workflows/{workflow_name}/facts/{fact_name}/ids"
        params = {"duration_fact": fact_name, "breakdown_filter": fact_name}
        params.update(OkahuSpanLoader._window_params(start_time, end_time))
        params.update(OkahuSpanLoader._eval_param(eval_filter))
        context_msg = f"{fact_name} ids in workflow '{workflow_name}'"

        def fetch(page_params):
            return OkahuSpanLoader._do_get(
                url, headers, params=page_params, timeout=timeout,
                context_msg=context_msg)

        def extract(envelope):
            """The ids of one page. fact_ids is keyed by id, so its keys ARE the
            ids and the order is the server's."""
            fact_ids = (envelope.get("fact_ids")
                        if isinstance(envelope, dict) else None)
            if isinstance(fact_ids, dict):
                return list(fact_ids)
            if isinstance(fact_ids, list):
                return [item.get("fact_id") if isinstance(item, dict) else item
                        for item in fact_ids]
            if fact_ids is None:
                return []
            raise ConnectionError(
                f"Okahu returned an unexpected 'fact_ids' for {fact_name} in "
                f"workflow '{workflow_name}': {type(fact_ids).__name__}")

        return OkahuSpanLoader._collect_paged(
            fetch, params, page_size, extract, context_msg)

    @staticmethod
    def get_trace_ids(
        workflow_name: str,
        fact_name: Optional[str] = None,
        fact_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        eval_filter: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> List[str]:
        """Fetch trace IDs from Okahu filtered by a fact.

        Uses:  GET /api/v1/workflows/<wf>/traces?duration_fact=<fact_name>&fact_ids=<fact_id>

        With no fact filter the query returns every trace in the workflow (or in
        the time window, when one is given) -- that is how a test-case set is
        enumerated. Pass both fact_name and fact_id to narrow to one fact.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            fact_name: The fact to filter by (e.g. ``agentic_session``). Optional,
                but only together with fact_id.
            fact_id: The fact value (e.g. a session ID). Optional, but only
                together with fact_name.
            eval_filter: Optional ``eval`` filter narrowing the result set to
                traces carrying it -- a bare eval name, or the API's
                ``name:label;name:label`` form.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds. Defaults to
                OKAHU_API_TIMEOUT, then ``DEFAULT_API_TIMEOUT`` (120).
            page_size: Rows per page. Defaults to DEFAULT_PAGE_SIZE (200);
                the server rejects anything above MAX_PAGE_SIZE (1000).

        Returns:
            A list of trace ID strings, across every page.

        Raises:
            ValueError: If exactly one of fact_name / fact_id is given. Half a
                filter is a mistake, not a mode.
        """
        if (fact_name is None) != (fact_id is None):
            raise ValueError(
                "fact_name and fact_id must be given together or not at all; "
                f"got fact_name={fact_name!r}, fact_id={fact_id!r}")

        page_size = OkahuSpanLoader._resolve_page_size(page_size)
        base = OkahuSpanLoader._get_api_base(endpoint)
        headers = OkahuSpanLoader._get_headers(api_key)
        params = {}
        if fact_name is not None:
            params["duration_fact"] = fact_name
            params["fact_ids"] = fact_id
        params.update(OkahuSpanLoader._window_params(start_time, end_time))
        params.update(OkahuSpanLoader._eval_param(eval_filter))
        context_msg = (f"traces for {fact_name}='{fact_id}' in workflow "
                       f"'{workflow_name}'" if fact_name
                       else f"traces in workflow '{workflow_name}'")

        def fetch(page_params):
            # _get_resource re-resolves the apps/workflows namespace per page.
            # When 'apps' is correct it returns on the first attempt, so this
            # costs nothing; only the fallback path spends one debug-level 404
            # per page, which is cheaper than threading resolved state through
            # the pager.
            return OkahuSpanLoader._get_resource(
                base, f"{workflow_name}/traces", headers, params=page_params,
                timeout=timeout, context_msg=context_msg)

        def extract(envelope):
            trace_list = OkahuSpanLoader._unwrap_list(
                envelope, ("traces", "data", "results"),
                context_msg=f"traces for {fact_name}='{fact_id}'")
            ids = []
            for item in trace_list:
                if isinstance(item, dict) and "trace_id" in item:
                    ids.append(item["trace_id"])
                elif isinstance(item, str):
                    ids.append(item)
            return ids

        trace_ids = OkahuSpanLoader._collect_paged(
            fetch, params, page_size, extract, context_msg)

        logger.debug(
            "Found %d trace(s) for %s='%s' in workflow '%s'",
            len(trace_ids), fact_name, fact_id, workflow_name,
        )
        return trace_ids

    @staticmethod
    def get_spans(
        workflow_name: str,
        trace_id: str,
        filter_fact: Optional[str] = None,
        filter_fact_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[ReadableSpan]:
        """Fetch spans from Okahu for a given trace_id.

        Uses:  GET /api/v1/workflows/<wf>/traces/<trace_id>/spans
        Optionally appends ``?filter_fact=<fact>&filter_fact_id=<id>``
        to filter spans server-side (e.g. by session).

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            trace_id: The trace ID (hex string) to fetch spans for.
            filter_fact: Optional server-side span filter fact name.
            filter_fact_id: Optional server-side span filter fact value.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds. Defaults to
                OKAHU_API_TIMEOUT, then ``DEFAULT_API_TIMEOUT`` (120).

        Returns:
            A list of ReadableSpan instances.

        Raises:
            ValueError: If OKAHU_API_KEY is not configured.
            ConnectionError: If the request to Okahu fails.
        """
        # Strip 0x prefix if present
        trace_id = trace_id.replace("0x", "")

        base = OkahuSpanLoader._get_api_base(endpoint)
        headers = OkahuSpanLoader._get_headers(api_key)
        params = {}
        if filter_fact and filter_fact_id:
            params["filter_fact"] = filter_fact
            params["filter_fact_id"] = filter_fact_id
        params.update(OkahuSpanLoader._window_params(start_time, end_time))

        span_data_list = OkahuSpanLoader._get_resource(
            base, f"{workflow_name}/traces/{trace_id}/spans", headers,
            params=params or None, timeout=timeout,
            context_msg=f"spans for trace_id '{trace_id}' in workflow '{workflow_name}'"
        )

        span_data_list = OkahuSpanLoader._unwrap_list(
            span_data_list, ("spans", "batch", "data", "results", "trace_spans"),
            context_msg=f"spans for trace_id '{trace_id}'"
        )

        span_list = []
        for item in span_data_list:
            span = JSONSpanLoader._from_dict(span_data=item)
            span_list.append(span)
        # verify that there's a span with span.attributes["span.type"] == "workflow" otherwise raise HttpError 404
        if not any(span.attributes.get("span.type") == "workflow" for span in span_list):
            raise requests.HTTPError(f"No workflow span found in trace '{trace_id}' - possible invalid trace ID or trace not fully ingested yet.")

        logger.debug("Loaded %d spans from Okahu for trace_id '%s'", len(span_list), trace_id)
        return span_list

    @staticmethod
    def load_by_session(
        workflow_name: str,
        session_id: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> List[ReadableSpan]:
        """Fetch all spans for every trace in a session.

        This is a convenience wrapper around ``load_by_scope()`` that uses
        the standard "agent_sessions" scope name.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            session_id: The agent session ID.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds. Defaults to
                OKAHU_API_TIMEOUT, then ``DEFAULT_API_TIMEOUT`` (120).

        Returns:
            A flat list of ReadableSpan instances from all matching traces.

        Raises:
            ConnectionError: If no traces found or API call fails.
        """
        return OkahuSpanLoader.load_by_scope(
            workflow_name=workflow_name,
            scope_name=OkahuSpanLoader.AGENT_SESSIONS_SCOPE,
            scope_id=session_id,
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
        )

    @staticmethod
    def load_by_scope(
        workflow_name: str,
        scope_name: str,
        scope_id: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[ReadableSpan]:
        """Fetch all spans for every trace matching a custom scope.

        This is a generic method that works with any Okahu fact/scope.
        For example:
        - scope_name="agent_sessions", scope_id="session_123"
        - scope_name="test_id", scope_id="test_456"
        - scope_name="my_custom_scope", scope_id="custom_789"

        1. GET traces with ``duration_fact=<scope_name>&fact_ids=<scope_id>``
        2. For each trace, GET spans with ``filter_fact=<scope_name>&filter_fact_id=<scope_id>``
        3. Return ReadableSpan objects.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            scope_name: The name of the scope/fact to filter by.
            scope_id: The scope/fact value (e.g., session ID, test ID, etc.).
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds. Defaults to
                OKAHU_API_TIMEOUT, then ``DEFAULT_API_TIMEOUT`` (120).

        Returns:
            A flat list of ReadableSpan instances from all matching traces.

        Raises:
            ValueError: If scope_name or scope_id is empty.
            ConnectionError: If no traces found or API call fails.
        """
        # Validate inputs
        if not scope_name or not scope_name.strip():
            raise ValueError("scope_name cannot be empty")
        if not scope_id or not scope_id.strip():
            raise ValueError("scope_id cannot be empty")

        trace_ids = OkahuSpanLoader.get_trace_ids(
            workflow_name,
            fact_name=scope_name,
            fact_id=scope_id,
            endpoint=endpoint, api_key=api_key, timeout=timeout,
            start_time=start_time, end_time=end_time,
        )
        if not trace_ids:
            raise ConnectionError(
                f"No traces found for {scope_name}='{scope_id}' in workflow '{workflow_name}'"
            )

        all_spans: List[ReadableSpan] = []
        for tid in trace_ids:
            spans = OkahuSpanLoader.get_spans(
                workflow_name, tid,
                filter_fact=scope_name,
                filter_fact_id=scope_id,
                endpoint=endpoint, api_key=api_key, timeout=timeout,
                start_time=start_time, end_time=end_time,
            )
            all_spans.extend(spans)

        logger.debug(
            "Loaded %d total spans across %d trace(s) for %s='%s'",
            len(all_spans), len(trace_ids), scope_name, scope_id,
        )
        return all_spans
