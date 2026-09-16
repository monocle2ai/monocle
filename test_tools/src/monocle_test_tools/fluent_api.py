from functools import wraps
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union
from monocle_apptrace.instrumentation.common.method_wrappers import monocle_trace_method
from monocle_apptrace.instrumentation.common.utils import get_workflow_name
from monocle_test_tools import eval_matrix
from monocle_test_tools.constants import CUSTOM_EVAL_TYPE
from monocle_test_tools.evals.okahu_filtered_eval import build_filtered_report
from monocle_test_tools.schema import Evaluation, FactID
from monocle_test_tools.span_loader import JSONSpanLoader, OkahuSpanLoader
from monocle_test_tools.testcase import Agent, FluentTestCase, Tool
from monocle_test_tools.testcase_args import (factid_import_kwargs, resolve_testcase,
                                              turn_inputs_from_spans)
from .comparer.comparer_manager import get_comparer
from .comparer.base_comparer import BaseComparer
from .comparer.default_comparer import DefaultComparer
from .comparer.token_match_comparer import TokenMatchComparer
from .evals.eval_manager import get_evaluator
from .evals.base_eval import BaseEval
from .validator import MonocleValidator
from .trace_utils import get_function_signature, get_caller_file_line
from .schema import MockTool
from opentelemetry.sdk.trace import Span

def setup_test_cases(source:str = "okahu", **kwargs) -> list[FluentTestCase]:
    """Build FluentTestCases from evals already recorded on a trace source.

    Turns a recorded population into a parametrizable list: each returned case
    points at one fact and carries the evals recorded on it as the expected
    results, ready to hand to ``with_trace_source(testcase=...)`` /
    ``run_agent(testcase=...)`` and ``check_eval(testcase=...)``.

    Args:
        source: Where the cases come from:
            - ``"okahu"`` (default) -- discover them from the evals recorded in
              the Okahu eval store.
            - ``"local"`` -- load a committed JSON array from ``path``.
        eval_name: Restrict to a single eval. Omitted means every eval recorded.
            Only meaningful for okahu, where it filters the query.
        **kwargs: Passed to the source. For okahu: workflow_name, start_time and
            end_time (all required), plus optional fact_name, category and
            page_size and compare_eval -- see OkahuSpanLoader.setup_test_cases.
            For local: ``path``.

    Returns:
        For okahu, one FluentTestCase per fact that has at least one labelled
        eval; for local, one per element of the file, in file order.

    Raises:
        ValueError: If *source* is unsupported, if ``path`` is missing or given
            alongside eval_name for the local source, or if the file is malformed.
        FileNotFoundError: If the local ``path`` does not exist.

    Example:
        CASES = setup_test_cases(source="okahu", workflow_name="wf",
                               start_time="2026-05-01", end_time="2026-06-30")

        @pytest.mark.parametrize("testcase", CASES)
        def test_regression(monocle_trace_asserter, testcase):
            monocle_trace_asserter.with_trace_source(testcase=testcase,
                                                     workflow_name="wf")
            monocle_trace_asserter.check_eval(testcase=testcase)

        # Freeze that set once, then re-run it with no network call:
        #   json.dump([c.model_dump() for c in CASES], open("cases.json", "w"))
        CASES = setup_test_cases(source="local", path="cases.json")
    """
    if source == "local":
        path = kwargs.pop("path", None)
        if not path:
            raise ValueError(
                "'path' is required for source='local'; it names the JSON file "
                "holding the array of test cases.")
        if kwargs:
            raise ValueError(
                f"source='local' takes only 'path'; got {sorted(kwargs)}")
        from .testcase import load_test_cases_from_json
        return load_test_cases_from_json(path)

    if source != "okahu":
        raise ValueError(
            f"setup_test_cases does not support source '{source}'; supported sources "
            "are 'okahu' (discover from the eval store) and 'local' (a JSON file).")
    from .okahu_span_loader import OkahuSpanLoader
    return OkahuSpanLoader.setup_test_cases(**kwargs)

# Fluent methods that select entities, and the kind of entity each selects. A
# testcase-driven chain may use only one kind, since each builds its own map.
_SELECTOR_KINDS = {"called_agent": "agent", "called_tool": "tool"}


def _entity_key(entity) -> tuple:
    """What makes two test-case entries the same queue.

    An Agent has no calling agent of its own, so agents key on the name alone --
    two same-name entries describe one agent and share its spans. A Tool keys on
    the name AND its caller, because from_spans records that caller and the same
    tool called by two agents is two different span sets. An entry naming no
    agent keys on the name alone and matches any caller.
    """
    agent = getattr(entity, "agent", None)
    return (entity.name, agent.name if agent else None)

def collect_assertions(func):
    """
        A decorator to collect assertion errors from fluent API methods. This supresses the AssertionError and collects all the assertions
        to be reported later. This also creates a new TraceAssertion instance to be returned for the next call in the fluent chain.
        This ensures the original span list is not overwritten by subsequent calls in the chain. 
    """
    @wraps(func)
    def decorator(asserter, *args, **kwargs):
        """Decorator to collect assertion errors from fluent API methods."""
        func_signature = get_function_signature(func, *args, **kwargs)
        fluent_chain:list[str] = []
        if len(asserter.fluent_chain) == 0:
            # add caller file and line number for the first call in the chain only
            func_signature = get_caller_file_line() + func_signature
        for signature in asserter.fluent_chain:
            fluent_chain.append(signature)
        fluent_chain.append(func_signature)

        # Chain-mixing guard. A chain either drives its assertions from a test
        # case or spells them out, never both -- a half-driven chain reads as if
        # the test case were being validated when most of it is being ignored.
        # The mode is set by the first decorated call and carried forward; it
        # lives on the derived asserter, so an independent chain starting from
        # the same fixture asserter is unaffected.
        testcase_given = kwargs.get("testcase") is not None
        if asserter.fluent_chain and asserter._testcase_mode is not None:
            if asserter._testcase_mode and not testcase_given:
                raise ValueError(
                    f"this chain is driven by a testcase; pass testcase= to "
                    f"{func.__name__}() as well, or start a new chain")
            if not asserter._testcase_mode and testcase_given:
                raise ValueError(
                    f"this chain does not use a testcase; drop testcase= from "
                    f"{func.__name__}(), or start a new chain")
        testcase_mode = (asserter._testcase_mode if asserter.fluent_chain
                         else testcase_given)

        # One selector kind per testcase chain. Each testcase-driven selector
        # builds its own entity map, so mixing two is ambiguous. Deliberately
        # scoped to testcase mode: called_agent("A").called_tool("T") is the
        # documented narrowing pattern and must keep working.
        selector = _SELECTOR_KINDS.get(func.__name__)
        testcase_selector = asserter._testcase_selector
        if testcase_given and selector is not None:
            if testcase_selector is not None and testcase_selector != selector:
                raise ValueError(
                    f"this chain already selects by {testcase_selector}; "
                    f"{func.__name__}() selects by {selector}, and a testcase chain "
                    "may use only one selector kind. Start a new chain.")
            testcase_selector = selector

        asserter = TraceAssertion(filtered_spans=asserter._filtered_spans, fluent_chain=fluent_chain,
                            is_assertion_failed=asserter.is_assertion_failed, _eval=asserter._eval,
                            okahu_filter=getattr(asserter, "_okahu_filter", None),
                            testcase_mode=testcase_mode,
                            entity_spans=asserter._entity_spans,
                            testcase_selector=testcase_selector)
        try:
            func(asserter, *args, **kwargs)
        except AssertionError as e:
            asserter.record_assertion(e, fluent_chain)
        return asserter
    return decorator

class TraceAssertion():
    
    """Fluent API for asserting properties on Monocle traces."""
    _eval:Optional[Union[str, BaseEval]]  = None
    _comparer: Union[str, BaseComparer] = DefaultComparer()
    _assertion_errors: list[dict[str, any]] = []
    # Per-test eval-result-matrix stash (see check_eval). Class-level like
    # _assertion_errors above: @collect_assertions returns a *new*
    # TraceAssertion for every fluent call, so instance attributes set
    # inside a chained method are invisible to the fixture's original
    # asserter. Reassign only via `TraceAssertion._last_eval = ...`
    # (class-qualified) and otherwise mutate in place (`.update(...)`) so
    # the fixture's `traceAssertion.__last_eval` sees the same object.
    _last_eval: Optional[dict[str, Any]] = None
    # Every eval run this test performed, oldest first. `_last_eval` is the final
    # entry; this list exists because one check_eval call can now run several
    # evals, and the results matrix must show all of them rather than the last.
    # Class-level for the same reason `_last_eval` is -- see the note above.
    _eval_stashes: list[dict[str, Any]] = []
    # Filter scope recorded by with_trace_source("okahu", start_time=..., end_time=...)
    # for eval-only filtered runs. Threaded through @collect_assertions like
    # _filtered_spans so the (decorated) check_eval can read it.
    _okahu_filter: Optional[dict] = None
    # Whether this chain drives its assertions from a testcase. Set by the first
    # decorated call and threaded per-chain by @collect_assertions; declared here
    # like _okahu_filter so an instance built outside __init__ still reads None.
    _testcase_mode: Optional[bool] = None
    # Spans of each entity a testcase-driven selector matched, in test-case order,
    # one pair per DISTINCT name. A list of pairs rather than a dict because Agent
    # is a pydantic model and therefore unhashable. Threaded per-chain by
    # @collect_assertions; class-level so an instance built outside __init__ reads None.
    _entity_spans: Optional[list] = None
    # Which selector kind opened this testcase chain ("agent" / "tool"). Two
    # different kinds in one chain would mean two entity maps; see collect_assertions.
    _testcase_selector: Optional[str] = None
    # Uniform eval report (filter mode = N facts; span mode = 1 fact). Class-scoped
    # like _last_eval so accessors on the fixture's original asserter can read it.
    _eval_report: Optional[dict] = None

    @staticmethod
    def get_trace_asserter():
        traceAssertion = TraceAssertion()
        traceAssertion.cleanup()
        return traceAssertion

    def __init__(self, filtered_spans:Optional[list[Span]] = None, fluent_chain:list[str] = []
                ,is_assertion_failed:bool = False, _eval:Optional[Union[str, BaseEval]] = None,
                okahu_filter:Optional[dict] = None,
                testcase_mode:Optional[bool] = None,
                *,
                entity_spans:Optional[list] = None,
                testcase_selector:Optional[str] = None) -> None:
        self._eval:Union[str, BaseEval]  = _eval
        self.validator = MonocleValidator()
        if filtered_spans is None:
            if self.validator.spans is not None and len(self.validator.spans) > 0:
                filtered_spans = self.validator.spans
        self._filtered_spans = filtered_spans
        self.fluent_chain = fluent_chain
        self.is_assertion_failed = is_assertion_failed
        self._skip_export = False
        self.mock_tools: Optional[list[MockTool]] = []
        self._okahu_filter = okahu_filter
        # None until the chain's first decorated call decides. See collect_assertions.
        self._testcase_mode = testcase_mode
        # Populated by a testcase-driven selector; read by the I/O assertions.
        self._entity_spans = entity_spans
        self._testcase_selector = testcase_selector
        
    def record_assertion(self, e:AssertionError, fluent_chain:list[str]) -> None:
        """Record an assertion error with its fluent chain context."""
        if self.is_assertion_failed == False:
            assertion_msg = e.args[0] if e.args else "Assertion failed"
            self._assertion_errors.append({"message": assertion_msg, "fluent_chain": fluent_chain})
            self.is_assertion_failed = True
        else:
            self._assertion_errors[-1]["fluent_chain"] = fluent_chain
    
    def get_assertion_messages(self) -> str:
        """Compile all assertion error messages into a single string."""
        assertion_message = f"Trace assertions : {len(self._assertion_errors)} failures:"
        for assertion in self._assertion_errors:
            assertion_message += f"{os.linesep}  " + assertion["message"] + " -> " + ".".join(assertion["fluent_chain"])
        return assertion_message

    def has_assertions(self) -> bool:
        return len(self._assertion_errors) > 0

    @property
    def assertions(self) -> list[AssertionError]:
        return self._assertion_errors

    def cleanup(self) -> None:
        """Cleanup validator state and evaluation resources."""
        # Clean up evaluation resources (e.g., delete traces from eval service)
        if self._eval is not None and hasattr(self._eval, 'cleanup'):
            try:
                self._eval.cleanup()
            except Exception:
                pass
        
        # Clean up validator state
        self.validator.cleanup()
        self._filtered_spans = None
        self._okahu_filter = None
        TraceAssertion._assertion_errors = []
        TraceAssertion._last_eval = None
        TraceAssertion._eval_stashes = []
        TraceAssertion._entity_spans = None
        TraceAssertion._testcase_selector = None
        TraceAssertion._eval_report = None

    @staticmethod
    def _validate_count_params(count: Optional[int], min_count: Optional[int], max_count: Optional[int]) -> None:
        """Validate that count parameters are not conflicting."""
        if count is not None and (min_count is not None or max_count is not None):
            raise ValueError("Cannot specify both 'count' and 'min_count'/'max_count'")

    def _check_aggregate_count(self, spans: list[Span], entity_type: str, count: Optional[int],
                                min_count: Optional[int], max_count: Optional[int], message: Optional[str]) -> None:
        """Helper to check count constraints for aggregate methods."""
        actual_count = len(spans)
        
        if count is not None or min_count is not None or max_count is not None:
            if count is not None and actual_count != count:
                raise AssertionError(message or f"Found {actual_count} total {entity_type} invocations, expected exactly {count}")
            if min_count is not None and actual_count < min_count:
                raise AssertionError(message or f"Found {actual_count} total {entity_type} invocations, expected at least {min_count}")
            if max_count is not None and actual_count > max_count:
                raise AssertionError(message or f"Found {actual_count} total {entity_type} invocations, expected at most {max_count}")
        else:
            if actual_count == 0:
                raise AssertionError(message or f"No {entity_type} invocations found")

    def _testcase_run_args(self, testcase:Union[FluentTestCase, dict], args:tuple) -> tuple:
        """The positional arguments a test case says to run the agent with.

        A FactID input means the run is a replay: the recorded trace is fetched
        only to recover the prompt it was driven with, so it is fetched with
        load_spans=False. Loading it would leave the *source* trace's spans and
        fact id on the validator, and the assertions that follow are about the
        new run, not the old one.

        workflow_name is deliberately not taken from the caller's kwargs -- those
        belong to the agent runner. The okahu fetch falls back to
        import_traces' own get_workflow_name(), which raises a clear error of its
        own when it cannot resolve.
        """
        testcase = resolve_testcase(testcase, args=args)
        if testcase.input is None:
            raise ValueError(
                f"testcase '{testcase.name}' has no input to run the agent with")
        if isinstance(testcase.input, FactID):
            spans = self.validator.import_traces(
                **factid_import_kwargs(testcase.input), load_spans=False)
            return turn_inputs_from_spans(spans)
        return tuple(testcase.input)

    def run_agent(self, agent, agent_type:str, *args, testcase:Optional[Union[FluentTestCase, dict]] = None, **kwargs) -> any:
        """Run the given agent with provided args and kwargs.

        Pass ``testcase`` instead of positional args to take the input from a
        FluentTestCase. When its input is a FactID, the recorded trace is loaded
        and the prompt it was run with is replayed against this agent.
        """
        if testcase is not None:
            args = self._testcase_run_args(testcase, args)
        return self.validator.run_agent(agent, agent_type, *args, mock_tools=self.mock_tools, **kwargs)

    async def run_agent_async(self, agent, agent_type:str, *args, session_id:str=None, turn_id:str=None, testcase:Optional[Union[FluentTestCase, dict]] = None, **kwargs) -> any:
        """Run the given async agent with provided args and kwargs.

        Pass ``turn_id`` to tag every span produced by this run with a
        ``scope.turn_id`` attribute. Pass ``testcase`` instead of positional args
        to take the input from a FluentTestCase, as ``run_agent`` does.
        """
        if testcase is not None:
            args = self._testcase_run_args(testcase, args)
        return await self.validator.run_agent_async(agent, agent_type, *args, session_id=session_id, turn_id=turn_id, mock_tools=self.mock_tools, **kwargs)

    def with_mock_tool(self, mock_tool:MockTool) -> 'TraceAssertion':
        """Set mock tools to be used during agent execution."""
        self.mock_tools.append(mock_tool)
        return self

    def with_evaluation(self, eval:Union[str, BaseEval], eval_options:Optional[dict] = {}) -> 'TraceAssertion':
        """Set the evaluation method for input/output comparisons."""
        updated_eval_options = eval_options.copy() if eval_options else {}
        updated_eval_options['trace_source'] = self.validator._trace_source
        self._eval = get_evaluator(eval, updated_eval_options)
        return self

    def with_comparer(self, comparer:Union[str, BaseComparer]) -> 'TraceAssertion':
        """Set the comparer for input/output comparisons."""
        self._comparer = get_comparer(comparer)
        return self

    def with_trace_source(self, source: Optional[str] = None,
                          testcase:Optional[Union[FluentTestCase, dict]] = None,
                          **kwargs) -> 'TraceAssertion':
        """Configure trace source for assertions.

        Args:
            source: Trace source type:
                - ``"local"`` (default) — Use traces from memory (current execution).
                - ``"file"`` — Load traces from local .monocle/*.json files.
                - ``"okahu"`` — Fetch traces from Okahu cloud.
            testcase: A FluentTestCase (or a dict in any shape it accepts) whose
                input is a FactID. It supplies ``id``, ``fact_name`` and
                ``scope_name``, which therefore may not also be passed.
                ``source`` and ``workflow_name`` remain explicit configuration
                and may accompany it; ``source`` falls back to the FactID's own.
            **kwargs: Additional arguments passed to ``import_traces()`` when
                source is "file" or "okahu". Common arguments:
                - id (str): Trace/session/scope ID
                - fact_name (str): "trace", "session", or "scope"
                - scope_name (str): Custom scope name (when fact_name="scope")
                - workflow_name (str): Okahu workflow name (required for "okahu")

        Returns:
            self for fluent chaining.

        Examples:
            # Use local/memory traces (default behavior)
            asserter.with_trace_source("local").called_tool("search")

            # Load from file
            asserter.with_trace_source(
                "file",
                id="abc123"
            ).called_tool("search")

            # Load from Okahu by session
            asserter.with_trace_source(
                "okahu",
                id="session_123",
                fact_name="session",
                workflow_name="my_app"
            ).called_tool("search")

            # Load from Okahu by custom scope
            asserter.with_trace_source(
                "okahu",
                id="test_456",
                fact_name="scope",
                scope_name="test_id",
                workflow_name="my_app"
            ).called_tool("search")
        """
        if testcase is not None:
            # Only the *identifying* arguments conflict: source and workflow_name
            # stay explicit configuration a test case does not carry.
            testcase = resolve_testcase(
                testcase, id=kwargs.get("id"), fact_name=kwargs.get("fact_name"),
                scope_name=kwargs.get("scope_name"))
            if not isinstance(testcase.input, FactID):
                raise ValueError(
                    f"with_trace_source needs a FactID input to load from; testcase "
                    f"'{testcase.name}' has {type(testcase.input).__name__}")
            kwargs.update(factid_import_kwargs(testcase.input))
            source = source or kwargs.pop("trace_source")
            kwargs.pop("trace_source", None)

        if source is None:
            source = "local"

        window_kwargs = ("start_time", "end_time")
        has_window = any(kwargs.get(k) is not None for k in window_kwargs)

        if source == "local":
            # Default behavior: use traces already in memory.
            if has_window:
                raise ValueError("Time-window filtering is only supported for source='okahu'.")
        elif source == "file":
            if has_window:
                raise ValueError("Time-window filtering is only supported for source='okahu'.")
            self.validator.import_traces(trace_source=source, **kwargs)
        elif source == "okahu":
            has_id = kwargs.get("id") is not None
            if has_window and has_id:
                # A window *with* an id is a bounded lookup: import that fact's
                # spans, narrowing the server-side query to the window. That is
                # not filter mode -- filter mode is eval-only and imports no
                # spans, which is what a window on its own selects.
                self.validator.import_traces(trace_source=source, **kwargs)
            elif has_window:
                # Filter mode: eval-only. Record the scope; import no spans.
                start_time, end_time = kwargs.get("start_time"), kwargs.get("end_time")
                if start_time is None or end_time is None:
                    raise ValueError("Filter mode requires both 'start_time' and 'end_time'.")
                workflow_name = kwargs.get("workflow_name")
                if not workflow_name:
                    raise ValueError("Filter mode requires 'workflow_name'.")
                workflows = ([workflow_name] if isinstance(workflow_name, str)
                             else list(workflow_name))
                self._okahu_filter = {"workflows": workflows, "start_time": start_time,
                                      "end_time": end_time,
                                      "fact_name": kwargs.get("fact_name", "traces")}
            else:
                # Direct id mode (unchanged): single id imported into memory.
                self.validator.import_traces(trace_source=source, **kwargs)
        else:
            raise ValueError(
                f"Unsupported trace source: '{source}'. "
                "Supported sources: 'local', 'file', 'okahu'."
            )

        return self

    @collect_assertions
    def called_tool(self, tool_name:Optional[str] = None, agent_name:Optional[str] = None, count:Optional[int] = None,
                    min_count:Optional[int] = None, max_count:Optional[int] = None, message:Optional[str] = None,
                    *,
                    testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert tool invocation with optional agent filter and count constraints (count, min_count, max_count).

        Args:
            testcase: Assert every tool the test case names instead of one, and
                record each one's spans for the input/output checks that follow.
                Cannot be combined with
                tool_name/agent_name/count/min_count/max_count.
        """
        if testcase is not None:
            return self._called_tool_testcase(
                testcase, tool_name=tool_name, agent_name=agent_name, count=count,
                min_count=min_count, max_count=max_count, message=message)
        if tool_name is None:
            raise ValueError("tool_name is required")
        TraceAssertion._validate_count_params(count, min_count, max_count)
        self._filtered_spans = self.validator._get_tool_invocation_spans(tool_name, agent_name, filtered_spans=self._filtered_spans)
        actual_count = len(self._filtered_spans)
        
        if count is not None or min_count is not None or max_count is not None:
            entity_prefix = f"Tool '{tool_name}' was called by agent '{agent_name}'" if agent_name else f"Tool '{tool_name}' was called"
            if count is not None and actual_count != count:
                raise AssertionError(message or f"{entity_prefix} {actual_count} times, expected exactly {count}")
            if min_count is not None and actual_count < min_count:
                raise AssertionError(message or f"{entity_prefix} {actual_count} times, expected at least {min_count}")
            if max_count is not None and actual_count > max_count:
                raise AssertionError(message or f"{entity_prefix} {actual_count} times, expected at most {max_count}")
        else:
            not_called_msg = f"Tool '{tool_name}' was not called by agent '{agent_name}'" if agent_name else f"Tool '{tool_name}' was not called"
            TraceAssertion._assert_on_spans(self._filtered_spans, not_called_msg, custom_message=message)
        return self

    @collect_assertions
    def does_not_call_tool(self, tool_names:str, agent_name:Optional[str] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the given tool was not called, optionally by a specific agent."""
        _filtered_spans = self.validator._get_tool_invocation_spans(tool_names, agent_name, filtered_spans=self._filtered_spans)
        if agent_name:
            TraceAssertion._assert_on_spans(_filtered_spans, f"Tool '{tool_names}' was called by agent '{agent_name}'", positive_test=False, custom_message=message)
        else:
            TraceAssertion._assert_on_spans(_filtered_spans, f"Tool '{tool_names}' was called", positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def called_agent(self, agent_name:Optional[str] = None, count:Optional[int] = None, min_count:Optional[int] = None, 
                     max_count:Optional[int] = None, message:Optional[str] = None,
                     *,
                     testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert agent invocation with optional count constraints (count, min_count, max_count).

        Args:
            testcase: Assert every agent the test case names instead of one, and
                record each one's spans for the input/output checks that follow.
                Cannot be combined with agent_name/count/min_count/max_count.
        """
        if testcase is not None:
            return self._called_agent_testcase(
                testcase, agent_name=agent_name, count=count,
                min_count=min_count, max_count=max_count, message=message)
        if agent_name is None:
            raise ValueError("agent_name is required without a testcase")
        TraceAssertion._validate_count_params(count, min_count, max_count)
        self._filtered_spans = self.validator._get_agent_invocation_spans(agent_name, filtered_spans=self._filtered_spans)
        actual_count = len(self._filtered_spans)
        
        if count is not None or min_count is not None or max_count is not None:
            if count is not None and actual_count != count:
                raise AssertionError(message or f"Agent '{agent_name}' was called {actual_count} times, expected exactly {count}")
            if min_count is not None and actual_count < min_count:
                raise AssertionError(message or f"Agent '{agent_name}' was called {actual_count} times, expected at least {min_count}")
            if max_count is not None and actual_count > max_count:
                raise AssertionError(message or f"Agent '{agent_name}' was called {actual_count} times, expected at most {max_count}")
        else:
            TraceAssertion._assert_on_spans(self._filtered_spans, f"Agent '{agent_name}' was not called", custom_message=message)
        return self

    def _called_agent_testcase(self, testcase, *, agent_name, count, min_count,
                              max_count, message) -> 'TraceAssertion':
        """Resolve every agent a test case names into the entity-span map.

        One entry per DISTINCT name: from_spans keeps same-name agents with
        different input/output as separate entries, so a discovered test case
        routinely names one agent twice. Both entries describe one agent whose
        spans are one set, so the map holds it once and the per-entry
        expectations are checked against that shared list by the I/O assertions.

        Every missing agent is reported in a single AssertionError, because
        record_assertion keeps only the first failure of a chain.
        """
        testcase = resolve_testcase(testcase, agent_name=agent_name, count=count,
                                    min_count=min_count, max_count=max_count)
        if not testcase.agents:
            raise ValueError(
                f"testcase '{testcase.name}' names no agents to select; a selector "
                "with nothing to select must not read as a passing test")

        entity_spans, missing, matched = [], [], []
        for name in dict.fromkeys(agent.name for agent in testcase.agents):
            spans = self.validator._get_agent_invocation_spans(
                name, filtered_spans=self._filtered_spans)
            if spans:
                entity_spans.append((Agent(name=name), spans))
                matched.extend(spans)
            else:
                missing.append(name)

        self._entity_spans = entity_spans
        self._filtered_spans = matched

        if missing:
            raise AssertionError(message or (
                f"{len(missing)} of {len(missing) + len(entity_spans)} agents named by "
                f"testcase '{testcase.name}' were not called: " + ", ".join(
                    f"'{name}'" for name in missing)))
        return self

    def _called_tool_testcase(self, testcase, *, tool_name, agent_name, count,
                             min_count, max_count, message) -> 'TraceAssertion':
        """Resolve every tool a test case names into the entity-span map.

        Keyed by tool AND calling agent, unlike agents which key on the name
        alone: from_spans records the caller, and the same tool called by two
        agents is two different span sets. An entry naming no agent matches any
        caller.

        Every missing tool is reported in a single AssertionError, because
        record_assertion keeps only the first failure of a chain.
        """
        testcase = resolve_testcase(testcase, tool_name=tool_name,
                                    agent_name=agent_name, count=count,
                                    min_count=min_count, max_count=max_count)
        if not testcase.tools:
            raise ValueError(
                f"testcase '{testcase.name}' names no tools to select; a selector "
                "with nothing to select must not read as a passing test")

        entity_spans, missing, matched, seen = [], [], [], set()
        for tool in testcase.tools:
            key = _entity_key(tool)
            if key in seen:
                continue
            seen.add(key)
            name, caller = key
            spans = self.validator._get_tool_invocation_spans(
                name, caller, filtered_spans=self._filtered_spans)
            if spans:
                entity_spans.append((Tool(name=name, agent=tool.agent), spans))
                matched.extend(spans)
            else:
                missing.append(f"'{name}'" + (f" called by '{caller}'" if caller else ""))

        self._entity_spans = entity_spans
        self._filtered_spans = matched

        if missing:
            raise AssertionError(message or (
                f"{len(missing)} of {len(missing) + len(entity_spans)} tools named by "
                f"testcase '{testcase.name}' were not called: " + ", ".join(missing)))
        return self

    def _entity_span_list(self, entity) -> Optional[list]:
        """Spans matched for `entity`, or None when the selector did not match it."""
        key = _entity_key(entity)
        for matched, spans in self._entity_spans or []:
            if _entity_key(matched) == key:
                return spans
        return None

    @collect_assertions
    def does_not_call_agent(self, agent_name:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the given agent was not called."""
        _filtered_spans = self.validator._get_agent_invocation_spans(agent_name, filtered_spans=self._filtered_spans)
        TraceAssertion._assert_on_spans(_filtered_spans, f"Agent '{agent_name}' was called", positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def called_agents(self, count:Optional[int] = None, min_count:Optional[int] = None,
                      max_count:Optional[int] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert total agent invocations across all agents with count constraints (count, min_count, max_count)."""
        TraceAssertion._validate_count_params(count, min_count, max_count)
        agent_spans = self.validator._get_all_agent_invocation_spans(filtered_spans=self._filtered_spans)
        self._check_aggregate_count(agent_spans, "agent", count, min_count, max_count, message)
        return self

    @collect_assertions
    def called_tools(self, count:Optional[int] = None, min_count:Optional[int] = None,
                     max_count:Optional[int] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert total tool invocations across all tools with count constraints (count, min_count, max_count)."""
        TraceAssertion._validate_count_params(count, min_count, max_count)
        tool_spans = self.validator._get_all_tool_invocation_spans(filtered_spans=self._filtered_spans)
        self._check_aggregate_count(tool_spans, "tool", count, min_count, max_count, message)
        return self

    @collect_assertions
    def has_attribute(self, attribute_name:str, expected:Optional[any] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that a span carries the given attribute (optionally with a specific value).

        Filters the current spans down to those matching, so subsequent chained
        assertions operate on the matching subset. When ``expected`` is None, only the
        presence of the attribute is checked.
        """
        matching_spans = self._filter_spans_by_attribute(self._filtered_spans, attribute_name, expected)
        self._filtered_spans = matching_spans
        if not matching_spans:
            if message:
                raise AssertionError(message)
            if expected is None:
                raise AssertionError(f"No span found with attribute '{attribute_name}'")
            raise AssertionError(f"No span found with attribute '{attribute_name}' == '{expected}'")
        return self

    @collect_assertions
    def does_not_have_attribute(self, attribute_name:str, expected:Optional[any] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that no span carries the given attribute (optionally with a specific value)."""
        matching_spans = self._filter_spans_by_attribute(self._filtered_spans, attribute_name, expected)
        if matching_spans:
            if message:
                raise AssertionError(message)
            if expected is None:
                raise AssertionError(f"Span found with attribute '{attribute_name}', but was not expected")
            raise AssertionError(f"Span found with attribute '{attribute_name}' == '{expected}', but was not expected")
        return self

    @collect_assertions
    def has_event(self, event_name:str, attribute_name:Optional[str] = None,
                  expected:Optional[Any] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that a span has a named event and, optionally, a matching attribute.

        Filters the current spans to those containing the matching event, allowing
        subsequent fluent assertions to continue from the same spans. Values are
        compared without coercion; string values use the configured comparer.
        """
        matching_spans = self._filter_spans_by_event(
            self._filtered_spans, event_name, attribute_name, expected
        )
        self._filtered_spans = matching_spans
        if not matching_spans:
            if message:
                raise AssertionError(message)
            if attribute_name is None:
                raise AssertionError(f"No span found with event '{event_name}'")
            if expected is None:
                raise AssertionError(
                    f"No span found with event '{event_name}' containing attribute '{attribute_name}'"
                )
            raise AssertionError(
                f"No span found with event '{event_name}' containing attribute "
                f"'{attribute_name}' == '{expected}'"
            )
        return self

    @collect_assertions
    def where(self, attribute:Optional[dict] = None, event:Optional[dict] = None,
              predicate:Optional[Callable[[Span], bool]] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Generic span selector: narrow the filtered spans to those matching every given criterion.

        This is the generic building block behind the more specific selectors
        (``has_attribute``, ``has_event``). All provided criteria are AND-ed together;
        a span matches when:

          - every ``{name: expected}`` in ``attribute`` matches the span's attributes
            (``expected`` of None checks presence only; strings use the configured comparer),
          - it contains an event matching the ``event`` spec, and
          - ``predicate(span)`` is truthy.

        Args:
            attribute: Mapping of attribute name -> expected value (None = presence check).
            event: Event spec ``{"name": <event_name>, "attributes": {<attr>: <expected>}}``.
                ``"attributes"`` is optional (event-presence check when omitted); per-attribute
                ``expected`` of None checks presence only.
            predicate: Callable receiving a Span and returning a bool for arbitrary matching.
            message: Optional custom error message.

        Example:
            asserter.where(
                attribute={"span.type": "agentic.turn"},
                event={"name": "metadata", "attributes": {"total_tokens": 1000}},
            )
        """
        if attribute is None and event is None and predicate is None:
            raise ValueError("where() requires at least one of 'attribute', 'event', or 'predicate'.")
        matching_spans = self._filter_spans_where(self._filtered_spans, attribute, event, predicate)
        self._filtered_spans = matching_spans
        if not matching_spans:
            raise AssertionError(message or ("No span found matching " + self._describe_where(attribute, event, predicate)))
        return self

    @collect_assertions
    def does_not_match(self, attribute:Optional[dict] = None, event:Optional[dict] = None,
                       predicate:Optional[Callable[[Span], bool]] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Negative counterpart of ``where``: assert no span matches all given criteria.

        Accepts the same ``attribute``/``event``/``predicate`` criteria as ``where``.
        """
        if attribute is None and event is None and predicate is None:
            raise ValueError("does_not_match() requires at least one of 'attribute', 'event', or 'predicate'.")
        matching_spans = self._filter_spans_where(self._filtered_spans, attribute, event, predicate)
        if matching_spans:
            raise AssertionError(message or ("Span found matching " + self._describe_where(attribute, event, predicate)
                                             + ", but was not expected"))
        return self

    @staticmethod
    def _describe_where(attribute:Optional[dict], event:Optional[dict], predicate:Optional[Callable]) -> str:
        """Build a human-readable description of a where() criteria set for error messages."""
        parts = []
        if attribute is not None:
            parts.append(f"attribute(s) {attribute}")
        if event is not None:
            parts.append(f"event {event}")
        if predicate is not None:
            parts.append("predicate")
        return " and ".join(parts) if parts else "given criteria"

    def _value_matches(self, expected:Optional[Any], actual:Any) -> bool:
        """Return True when ``actual`` satisfies ``expected`` (None = present/any value)."""
        if expected is None:
            return True
        if isinstance(expected, str) and isinstance(actual, str):
            return self._comparer.compare(expected, actual)
        return actual == expected

    def _span_matches_attributes(self, span:Span, attribute:dict) -> bool:
        """Return True when the span carries every requested attribute with a matching value."""
        for name, expected in attribute.items():
            actual = span.attributes.get(name)
            if actual is None:
                return False
            if not self._value_matches(expected, actual):
                return False
        return True

    def _span_matches_event(self, span:Span, event:dict) -> bool:
        """Return True when the span has an event matching the ``event`` spec."""
        if not isinstance(event, dict):
            raise ValueError("'event' must be a dict, e.g. {'name': 'metadata', 'attributes': {...}}")
        event_name = event.get("name")
        if event_name is None:
            raise ValueError("'event' spec requires a 'name' key.")
        attr_spec = event.get("attributes") or {}
        for ev in getattr(span, "events", []) or []:
            if ev.name != event_name:
                continue
            if not attr_spec:
                return True
            ev_attrs = ev.attributes or {}
            if all(name in ev_attrs and self._value_matches(expected, ev_attrs[name])
                   for name, expected in attr_spec.items()):
                return True
        return False

    def _filter_spans_where(self, spans:Optional[Sequence[Span]], attribute:Optional[dict],
                            event:Optional[dict], predicate:Optional[Callable[[Span], bool]]) -> list[Span]:
        """Return spans satisfying all of the provided (attribute/event/predicate) criteria."""
        matching_spans = []
        for span in spans or []:
            if attribute is not None and not self._span_matches_attributes(span, attribute):
                continue
            if event is not None and not self._span_matches_event(span, event):
                continue
            if predicate is not None and not predicate(span):
                continue
            matching_spans.append(span)
        return matching_spans

    def _filter_spans_by_attribute(self, spans:Optional[list[Span]], attribute_name:str, expected:Optional[any]) -> list[Span]:
        """Return spans whose attribute ``attribute_name`` is present (and equals ``expected`` when given)."""
        return self._filter_spans_where(spans, {attribute_name: expected}, None, None)

    def _filter_spans_by_event(self, spans:Optional[Sequence[Span]], event_name:str,
                               attribute_name:Optional[str], expected:Optional[Any]) -> list[Span]:
        """Return spans containing an event that satisfies the requested attribute match."""
        event_spec:dict = {"name": event_name}
        if attribute_name is not None:
            event_spec["attributes"] = {attribute_name: expected}
        return self._filter_spans_where(spans, None, event_spec, None)

    @collect_assertions
    def has_input(self, expected_input:Optional[str] = None, message:Optional[str] = None,
                       *,
                       testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the input matches the expected input."""
        if testcase is not None:
            resolve_testcase(testcase, expected_input=expected_input)
            self._verify_io_testcase(testcase, field="input", comparer=self._comparer,
                                     positive_test=True, message=message)
            return self
        if expected_input is None:
            raise ValueError("expected_input is required without a testcase")
        self._verify_input_output(self._filtered_spans, expected_inputs=[expected_input],
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def has_any_input(self, *expected_inputs:str, message:Optional[str] = None, testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that any of the expected inputs match."""
        if testcase is not None:
            raise ValueError(
                "this method does not support 'testcase'; it takes any-of values, "
                "and an agent records exactly one input and one output. Use the "
                "singular form (e.g. contains_output) with a testcase.")
        if not expected_inputs:
            raise ValueError("At least one expected_input is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(expected_inputs),
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_input(self, unexpected_input:Optional[str] = None, message:Optional[str] = None,
                                 *,
                                 testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the input does not match the unexpected input."""
        if testcase is not None:
            resolve_testcase(testcase, unexpected_input=unexpected_input)
            self._verify_io_testcase(testcase, field="input", comparer=self._comparer,
                                     positive_test=False, message=message)
            return self
        if unexpected_input is None:
            raise ValueError("unexpected_input is required without a testcase")
        self._verify_input_output(self._filtered_spans, expected_inputs=[unexpected_input],
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_any_input(self, *unexpected_inputs:str, message:Optional[str] = None, testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that none of the unexpected inputs match."""
        if testcase is not None:
            raise ValueError(
                "this method does not support 'testcase'; it takes any-of values, "
                "and an agent records exactly one input and one output. Use the "
                "singular form (e.g. contains_output) with a testcase.")
        if not unexpected_inputs:
            raise ValueError("At least one unexpected_input is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(unexpected_inputs),
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def contains_input(self, expected_input_substring:Optional[str] = None, message:Optional[str] = None,
                            *,
                            testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the input contains the expected substring"""
        if testcase is not None:
            resolve_testcase(testcase, expected_input_substring=expected_input_substring)
            self._verify_io_testcase(testcase, field="input", comparer=TokenMatchComparer(),
                                     positive_test=True, message=message)
            return self
        if expected_input_substring is None:
            raise ValueError("expected_input_substring is required without a testcase")
        self._verify_input_output(self._filtered_spans, expected_inputs=[expected_input_substring],
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def contains_any_input(self, *expected_input_substrings:str, message:Optional[str] = None, testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that any input contains the expected substring"""
        if testcase is not None:
            raise ValueError(
                "this method does not support 'testcase'; it takes any-of values, "
                "and an agent records exactly one input and one output. Use the "
                "singular form (e.g. contains_output) with a testcase.")
        if not expected_input_substrings:
            raise ValueError("At least one expected_input_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(expected_input_substrings),
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_input(self, unexpected_input_substring:Optional[str] = None, message:Optional[str] = None,
                                    *,
                                    testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the input does not contain the given substring"""
        if testcase is not None:
            resolve_testcase(testcase, unexpected_input_substring=unexpected_input_substring)
            self._verify_io_testcase(testcase, field="input", comparer=TokenMatchComparer(),
                                     positive_test=False, message=message)
            return self
        if unexpected_input_substring is None:
            raise ValueError("unexpected_input_substring is required without a testcase")
        self._verify_input_output(self._filtered_spans, expected_inputs=[unexpected_input_substring],
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_any_input(self, *unexpected_input_substrings:str, message:Optional[str] = None, testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that no input contains the given substrings"""
        if testcase is not None:
            raise ValueError(
                "this method does not support 'testcase'; it takes any-of values, "
                "and an agent records exactly one input and one output. Use the "
                "singular form (e.g. contains_output) with a testcase.")
        if not unexpected_input_substrings:
            raise ValueError("At least one unexpected_input_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(unexpected_input_substrings),
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def has_output(self, expected_output:Optional[str] = None, message:Optional[str] = None,
                        *,
                        testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the output matches the expected output."""
        if testcase is not None:
            resolve_testcase(testcase, expected_output=expected_output)
            self._verify_io_testcase(testcase, field="output", comparer=self._comparer,
                                     positive_test=True, message=message)
            return self
        if expected_output is None:
            raise ValueError("expected_output is required without a testcase")
        self._verify_input_output(self._filtered_spans, expected_inputs=[], expected_outputs=[expected_output],
                                 comparer=self._comparer, eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def has_any_output(self, *expected_outputs:str, message:Optional[str] = None, testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the output matches any of the expected outputs."""
        if testcase is not None:
            raise ValueError(
                "this method does not support 'testcase'; it takes any-of values, "
                "and an agent records exactly one input and one output. Use the "
                "singular form (e.g. contains_output) with a testcase.")
        if not expected_outputs:
            raise ValueError("At least one expected_output is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                    expected_outputs=list(expected_outputs), comparer=self._comparer, eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_output(self, unexpected_output:Optional[str] = None, message:Optional[str] = None,
                                  *,
                                  testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the output does not have the given output."""
        if testcase is not None:
            resolve_testcase(testcase, unexpected_output=unexpected_output)
            self._verify_io_testcase(testcase, field="output", comparer=self._comparer,
                                     positive_test=False, message=message)
            return self
        if unexpected_output is None:
            raise ValueError("unexpected_output is required without a testcase")
        self._verify_input_output(self._filtered_spans, expected_inputs=[] , expected_outputs=[unexpected_output],
                                 comparer=self._comparer, eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_any_output(self, *unexpected_outputs:str, message:Optional[str] = None, testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the output does not have any of the given outputs."""
        if testcase is not None:
            raise ValueError(
                "this method does not support 'testcase'; it takes any-of values, "
                "and an agent records exactly one input and one output. Use the "
                "singular form (e.g. contains_output) with a testcase.")
        if not unexpected_outputs:
            raise ValueError("At least one unexpected_output is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                 expected_outputs=list(unexpected_outputs), comparer=self._comparer, eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def contains_output(self, expected_output_substring:Optional[str] = None, message:Optional[str] = None,
                             *,
                             testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the output contains the expected substring."""
        if testcase is not None:
            resolve_testcase(testcase, expected_output_substring=expected_output_substring)
            self._verify_io_testcase(testcase, field="output", comparer=TokenMatchComparer(),
                                     positive_test=True, message=message)
            return self
        if expected_output_substring is None:
            raise ValueError("expected_output_substring is required without a testcase")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=[expected_output_substring], comparer=TokenMatchComparer(), eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def contains_any_output(self, *expected_output_substrings:str, message:Optional[str] = None, testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that any output contains the expected substring."""
        if testcase is not None:
            raise ValueError(
                "this method does not support 'testcase'; it takes any-of values, "
                "and an agent records exactly one input and one output. Use the "
                "singular form (e.g. contains_output) with a testcase.")
        if not expected_output_substrings:
            raise ValueError("At least one expected_output_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=list(expected_output_substrings), comparer=TokenMatchComparer(), eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_output(self, unexpected_output_substring:Optional[str] = None, message:Optional[str] = None,
                                     *,
                                     testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that the output does not contain the given substring."""
        if testcase is not None:
            resolve_testcase(testcase, unexpected_output_substring=unexpected_output_substring)
            self._verify_io_testcase(testcase, field="output", comparer=TokenMatchComparer(),
                                     positive_test=False, message=message)
            return self
        if unexpected_output_substring is None:
            raise ValueError("unexpected_output_substring is required without a testcase")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=[unexpected_output_substring], comparer=TokenMatchComparer(), eval=self._eval,
                                positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_any_output(self, *unexpected_output_substrings:str, message:Optional[str] = None, testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Assert that no output contains the given substrings."""
        if testcase is not None:
            raise ValueError(
                "this method does not support 'testcase'; it takes any-of values, "
                "and an agent records exactly one input and one output. Use the "
                "singular form (e.g. contains_output) with a testcase.")
        if not unexpected_output_substrings:
            raise ValueError("At least one unexpected_output_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=list(unexpected_output_substrings), comparer=TokenMatchComparer(), eval=self._eval,
                                positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def has_scope(self, scope_name:str, expected_value:Optional[str] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that at least one filtered span has the specified scope.

        Args:
            scope_name: Name of the scope (e.g., 'tenant_id', 'subscriptionId')
            expected_value: Expected value for the scope. If omitted, only the
                presence of the scope is checked, regardless of its value.
            message: Optional custom error message

        Example:
            asserter.has_scope("tenant_id", "customer-123")  # value check
            asserter.has_scope("tenant_id")                   # existence check
        """
        expected_values = None if expected_value is None else [expected_value]
        self._verify_scope(self._filtered_spans, scope_name, expected_values,
                          comparer=self._comparer, positive_test=True, custom_message=message)
        return self

    @collect_assertions
    def has_any_scope(self, scope_name:str, *expected_values:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that spans have the specified scope with any of the expected values.
        
        Args:
            scope_name: Name of the scope (e.g., 'tenant_id')
            expected_values: One or more expected values for the scope
            message: Optional custom error message
            
        Example:
            asserter.has_any_scope("tenant_id", "customer-123", "customer-456")
        """
        if not expected_values:
            raise ValueError("At least one expected_value is required")
        self._verify_scope(self._filtered_spans, scope_name, list(expected_values),
                          comparer=self._comparer, positive_test=True, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_scope(self, scope_name:str, unexpected_value:Optional[str] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that no filtered span has the specified scope.

        Args:
            scope_name: Name of the scope (e.g., 'tenant_id')
            unexpected_value: Value that should not be present. If omitted, the
                scope must be entirely absent, regardless of its value.
            message: Optional custom error message

        Example:
            asserter.does_not_have_scope("tenant_id", "customer-999")  # value check
            asserter.does_not_have_scope("tenant_id")                   # absence check
        """
        unexpected_values = None if unexpected_value is None else [unexpected_value]
        self._verify_scope(self._filtered_spans, scope_name, unexpected_values,
                          comparer=self._comparer, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_any_scope(self, scope_name:str, *unexpected_values:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that spans do not have the specified scope with any of the given values.
        
        Args:
            scope_name: Name of the scope (e.g., 'tenant_id')
            unexpected_values: Values that should not be present
            message: Optional custom error message
            
        Example:
            asserter.does_not_have_any_scope("tenant_id", "customer-999", "customer-000")
        """
        if not unexpected_values:
            raise ValueError("At least one unexpected_value is required")
        self._verify_scope(self._filtered_spans, scope_name, list(unexpected_values),
                          comparer=self._comparer, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def contains_scope(self, scope_name:str, expected_substring:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the scope value contains the expected substring.
        
        Args:
            scope_name: Name of the scope (e.g., 'tenant_id')
            expected_substring: Substring that should be present in the scope value
            message: Optional custom error message
            
        Example:
            asserter.contains_scope("tenant_id", "customer")
        """
        self._verify_scope(self._filtered_spans, scope_name, [expected_substring],
                          comparer=TokenMatchComparer(), positive_test=True, custom_message=message)
        return self

    @collect_assertions
    def contains_any_scope(self, scope_name:str, *expected_substrings:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the scope value contains any of the expected substrings.
        
        Args:
            scope_name: Name of the scope (e.g., 'tenant_id')
            expected_substrings: Substrings to search for
            message: Optional custom error message
            
        Example:
            asserter.contains_any_scope("tenant_id", "customer", "client")
        """
        if not expected_substrings:
            raise ValueError("At least one expected_substring is required")
        self._verify_scope(self._filtered_spans, scope_name, list(expected_substrings),
                          comparer=TokenMatchComparer(), positive_test=True, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_scope(self, scope_name:str, unexpected_substring:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the scope value does not contain the given substring.
        
        Args:
            scope_name: Name of the scope (e.g., 'tenant_id')
            unexpected_substring: Substring that should not be present
            message: Optional custom error message
            
        Example:
            asserter.does_not_contain_scope("tenant_id", "admin")
        """
        self._verify_scope(self._filtered_spans, scope_name, [unexpected_substring],
                          comparer=TokenMatchComparer(), positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_any_scope(self, scope_name:str, *unexpected_substrings:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the scope value does not contain any of the given substrings.
        
        Args:
            scope_name: Name of the scope (e.g., 'tenant_id')
            unexpected_substrings: Substrings that should not be present
            message: Optional custom error message
            
        Example:
            asserter.does_not_contain_any_scope("tenant_id", "admin", "root")
        """
        if not unexpected_substrings:
            raise ValueError("At least one unexpected_substring is required")
        self._verify_scope(self._filtered_spans, scope_name, list(unexpected_substrings),
                          comparer=TokenMatchComparer(), positive_test=False, custom_message=message)
        return self

    def _apply_eval_type(self, eval_name, template_path, template):
        """Detect which kind of eval template ``eval_name`` holds, for check_eval.

        A path object or a path-like string (``.json`` suffix, path separator,
        ``./``, ``../``) is a custom eval template file, so it moves to
        ``template_path``; a bare name is a built-in template and stays in
        ``eval_name``. Strings are classified by the configured evaluator's own rules
        (``BaseEval.classify_eval_input``, which OkahuEval overrides).

        Returns the ``(eval_name, template_path)`` pair check_eval works with.
        """
        if not eval_name:
            return eval_name, template_path
        if isinstance(eval_name, os.PathLike):
            eval_type = CUSTOM_EVAL_TYPE
        else:
            eval_cls = type(self._eval) if isinstance(self._eval, BaseEval) else BaseEval
            eval_type, _ = eval_cls.classify_eval_input(eval_name)
        if eval_type != CUSTOM_EVAL_TYPE:
            return eval_name, template_path
        if template_path or template:
            raise ValueError(
                f"'eval_name' ({eval_name}) is a custom eval template path; do not also pass "
                "'template_path' or 'template'.")
        return None, eval_name

    @collect_assertions
    def check_eval(self, eval_name:Optional[Union[str, Path]] = None, expected:Optional[Union[str, list[str]]] = None, not_expected:Optional[Union[str, list[str]]] = None, fact_name:Optional[str] = "traces", message:Optional[str] = None, template_path:Optional[Union[str, Path]] = None, *, template:Optional[dict] = None, min_facts:int = 1, fail_threshold:int = 0, max_facts:Optional[int] = None, testcase:Optional[Union[FluentTestCase, dict]] = None) -> 'TraceAssertion':
        """Validate evaluation results for the current filtered spans.

        Provide exactly one of:
          - eval_name: the eval template to run — either the name of a standard Okahu
            eval template (e.g. "hallucination") or the path of a JSON file holding a
            custom eval template (a Path or a path-like string, e.g.
            "templates/my_eval.json"). Which one it is is detected from the value, so
            a path is handled exactly as if passed as ``template_path``.
          - template_path: filesystem path to a custom-template JSON file. The file
            is loaded and the parsed dict is sent to the eval service. Server-side
            validation errors (HTTP 400) surface as AssertionError with the prefix
            'Custom template validation failed: <reason>'.
          - template: an inline custom-template dict.

        When with_trace_source("okahu", start_time=..., end_time=...) has recorded a
        filter scope, this runs the filtered (async job) flow instead of the span
        path; min_facts/fail_threshold/max_facts apply only in that mode.

        Args:
            testcase: A FluentTestCase (or a dict in any shape it accepts) whose
                evals to run. Each eval's name selects the template, classified
                exactly as this method's own ``eval_name`` is: a bare name is an
                Okahu template, while a Path *or* a path-like string (``.json``
                suffix, a separator, ``./``, ``../``) is a custom-template file.
                The eval's result is the expected label. Cannot be combined with
                eval_name, expected, not_expected, template_path or template.
        """
        eval_name, template_path = self._apply_eval_type(eval_name, template_path, template)

        filter_scope = getattr(self, "_okahu_filter", None)
        if testcase is not None:
            if filter_scope is not None:
                raise ValueError(
                    "testcase= is not supported with a time-window (filtered) source: "
                    "a time window identifies no single fact for the test case to name.")
            return self._check_eval_testcase(
                testcase, eval_name=eval_name, expected=expected,
                not_expected=not_expected, template_path=template_path,
                template=template, fact_name=fact_name, message=message)

        if filter_scope is None:
            # Span mode: filter-only params must not be used.
            if min_facts != 1 or fail_threshold != 0 or max_facts is not None:
                raise ValueError(
                    "min_facts/fail_threshold/max_facts apply only to a time-window "
                    "(filtered) source; set start_time/end_time on with_trace_source('okahu', ...).")
        else:
            return self._check_eval_filtered(
                filter_scope, eval_name=eval_name, expected=expected,
                not_expected=not_expected, template_path=template_path, template=template,
                min_facts=min_facts, fail_threshold=fail_threshold, max_facts=max_facts,
                message=message)

        failures, fact_records = self._check_eval_one(
            eval_name=eval_name, expected=expected, not_expected=not_expected,
            fact_name=fact_name, template_path=template_path, template=template,
            message=message)
        TraceAssertion._eval_report = build_filtered_report(
            expected, not_expected, fact_records, job_id=None)
        self._raise_eval_failures(eval_name, failures, len(fact_records), message)
        return self

    def _check_eval_entry(self, entry, *, fact_name) -> tuple:
        """Run one eval named by a test case, and return its outcome.

        The name is classified the way check_eval classifies its own eval_name --
        through the evaluator's ``classify_eval_input`` via ``_apply_eval_type``,
        so a path object *and* a path-like string both resolve to a custom
        template file while a bare name stays a built-in.

        That matters because a test case usually arrives as JSON, which has no
        Path type: a custom template reaches us as a string, and testing only
        for ``pathlib.Path`` sent that string on as a template *name*, leaving
        the eval service asked for a template called "./evals/my_eval.json".
        """
        eval_name, template_path = self._apply_eval_type(entry.name, None, None)
        return self._check_eval_one(
            eval_name=eval_name, expected=entry.result, not_expected=None,
            fact_name=fact_name, template_path=template_path, template=None)

    def _check_eval_testcase(self, testcase, *, eval_name, expected, not_expected,
                             template_path, template, fact_name, message) -> 'TraceAssertion':
        """Run every eval a test case names, reporting all their failures at once."""
        testcase = resolve_testcase(testcase, eval_name=eval_name, expected=expected,
                                    not_expected=not_expected,
                                    template_path=template_path, template=template)
        if not testcase.evals:
            raise ValueError(
                f"testcase '{testcase.name}' has no evals to check; a test case with "
                "nothing to assert must not read as a passing test")

        failures, fact_records = [], []
        for entry in testcase.evals:
            entry_failures, entry_facts = self._check_eval_entry(
                entry, fact_name=fact_name)
            failures.extend(entry_failures)
            fact_records.extend(entry_facts)

        TraceAssertion._eval_report = build_filtered_report(
            [entry.result for entry in testcase.evals], None, fact_records, job_id=None)
        self._raise_eval_failures(testcase.name, failures, len(fact_records), message)
        return self

    @staticmethod
    def _raise_eval_failures(eval_name:str, failures:list[str], fact_count:int,
                             message:Optional[str]) -> None:
        """Raise one AssertionError covering every failure, or return quietly."""
        if not failures:
            return
        if message:
            raise AssertionError(message)
        if len(failures) == 1:
            raise AssertionError(failures[0])
        raise AssertionError(
            f"Evaluation '{eval_name}' failed for {len(failures)} of {fact_count} facts:"
            + "".join(f"{os.linesep}  - {failure}" for failure in failures))

    def _check_eval_one(self, *, eval_name, expected, not_expected, fact_name,
                        template_path, template, message=None) -> tuple:
        """Run one eval against the current spans and return its outcome.

        Returns rather than raises so a caller running several evals can report
        every failure: record_assertion only keeps the first AssertionError of a
        chain, so N raises would surface as one.

        Returns:
            ``(failures, fact_records)`` -- the failure messages, and one report
            record per graded fact for build_filtered_report.
        """
        if sum(bool(x) for x in (eval_name, template_path, template)) != 1:
            raise ValueError(
                "Provide exactly one of 'eval_name' (for Okahu templates), "
                "'template_path' (for a custom-template JSON file), or "
                "'template' (for an inline custom-template dict).")

        if template_path:
            path_obj = Path(template_path)
            if not path_obj.is_file():
                raise AssertionError(f"Custom template file not found: {template_path}")
            try:
                loaded = json.loads(path_obj.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"Custom template file is not valid JSON: {template_path} — {exc}"
                ) from exc
            # Accept either the inner template ({"name": ..., "eval_prompt": ..., ...})
            # or the API-request-body shape ({"template": {...inner...}}). Unwrap the
            # outer "template" key when present so evaluate() gets the inner dict —
            # evaluate() will re-wrap once for the HTTP payload.
            if (
                isinstance(loaded, dict)
                and set(loaded.keys()) == {"template"}
                and isinstance(loaded["template"], dict)
            ):
                template = loaded["template"]
            else:
                template = loaded
            eval_name = template.get("name", "custom_eval")
        elif template and not eval_name:
            eval_name = template.get("name", "custom_eval")

        if expected is None and not_expected is None:
            raise ValueError("At least one of 'expected' or 'not_expected' must be provided")
        positive = [expected] if isinstance(expected, str) else expected if expected is not None else []
        negative = [not_expected] if isinstance(not_expected, str) else not_expected if not_expected is not None else []

        if negative:
            overlap = set(positive) & set(negative)
            if overlap:
                raise ValueError(f"Overlapping evaluation results found in 'expected' and 'not_expected': {overlap}. Please ensure they are mutually exclusive.")

        if self._eval is None:
            raise AssertionError(message if message else "No evaluator configured. Call with_evaluation before check_eval.")
        if not self._filtered_spans:
            raise AssertionError(message if message else "No spans available for evaluation. Chain a span selector before check_eval.")

        # Stash a per-call eval-result-matrix record on the asserter (additive,
        # opt-in recorder in pytest_plugin.py reads this via `_last_eval`).
        # Populated up-front so trace_id/expected survive even if evaluate()
        # raises; updated below once a label/explanation are obtained.
        try:
            trace_id = format(self._filtered_spans[0].get_span_context().trace_id, "032x")
        except Exception:
            trace_id = ""
        # Class-qualified (not `self._last_eval = ...`): @collect_assertions
        # hands check_eval a fresh TraceAssertion instance per call, so a
        # plain `self._last_eval = ...` would only shadow this one
        # throwaway instance and never reach the fixture's original
        # asserter. See the class-attribute comment above.
        TraceAssertion._last_eval = {
            "trace_id": trace_id,
            "expected": expected,
            "fact_name": fact_name,
            "label": None,
            "explanation": "",
            "judge_output": {},
            "total_tokens": None,
        }
        # Appended, not replaced: one check_eval call can run several evals and
        # the results matrix must show every one of them.
        TraceAssertion._eval_stashes = TraceAssertion._eval_stashes + [TraceAssertion._last_eval]

        eval_result, explanation = self._eval.evaluate(filtered_spans=self._filtered_spans, eval_name=eval_name, fact_name=fact_name, template=template)

        self._last_eval.update(
            label=eval_result,
            explanation=explanation,
            judge_output=getattr(self._eval, "last_judge_output", {}) or {},
            total_tokens=getattr(self._eval, "last_total_tokens", None),
        )

        # Grade every fact the evaluator returned (one per turn/session when
        # fact_name resolves to many), not just the last one. Fall back to the
        # single returned label when the evaluator doesn't break results down by
        # fact. isinstance guards a MagicMock evaluator (truthy, but not a list).
        fact_results = getattr(self._eval, "last_fact_results", None)
        if isinstance(fact_results, list) and fact_results:
            facts = [(r["fact_id"], r["eval_result"]["label"],
                      r["eval_result"].get("explanation", "")) for r in fact_results]
        else:
            facts = [(trace_id, eval_result, explanation)]

        failures = []
        for fact_id, eval_result, explanation in facts:
            if positive and eval_result not in positive:
                failures.append(f"Evaluation '{eval_name}' did not match expected result for fact '{fact_id}'. Expected one of {positive}. Received '{eval_result}'. \n Explanation: {explanation}")
            elif negative and eval_result in negative:
                failures.append(f"Evaluation '{eval_name}' matched an unexpected result for fact '{fact_id}'. Should not be any of {negative}. Received '{eval_result}'. \n Explanation: {explanation}")

        return failures, [{"fact_id": fact_id, "job_id": None, "eval_found": True,
                           "eval_result": {"label": label, "explanation": explanation},
                           "workflow": ""}
                          for fact_id, label, explanation in facts]

    def _check_eval_filtered(self, scope, *, eval_name, expected, not_expected,
                             template_path, template, min_facts, fail_threshold,
                             max_facts, message):
        from monocle_test_tools.evals.okahu_filtered_eval import OkahuFilteredEval
        selectors = [s for s in (eval_name, template_path, template) if s]
        if len(selectors) != 1:
            raise ValueError("Provide exactly one of 'eval_name', 'template_path', or 'template'.")
        if isinstance(expected, dict):
            raise ValueError("check_eval takes 'expected' as a str/list, not a dict.")
        if expected is None and not_expected is None:
            raise ValueError("Filtered check_eval requires 'expected' and/or 'not_expected'.")
        acc = None if expected is None else ([expected] if isinstance(expected, str) else list(expected))
        neg = [] if not_expected is None else ([not_expected] if isinstance(not_expected, str) else list(not_expected))
        if acc and set(acc) & set(neg):
            raise ValueError(f"'expected' and 'not_expected' overlap: {set(acc) & set(neg)}.")

        if template_path:
            path_obj = Path(template_path)
            if not path_obj.is_file():
                raise AssertionError(f"Custom template file not found: {template_path}")
            loaded = json.loads(path_obj.read_text(encoding="utf-8"))
            if (isinstance(loaded, dict) and set(loaded.keys()) == {"template"}
                    and isinstance(loaded["template"], dict)):
                template = loaded["template"]
            else:
                template = loaded

        client = OkahuFilteredEval.from_env()
        report = client.run_filtered(
            scope["workflows"], accepted=expected, not_expected=not_expected,
            eval_name=eval_name, template=template, fact_name=scope["fact_name"],
            start_time=scope["start_time"], end_time=scope["end_time"],
            min_facts=min_facts, max_facts=max_facts)
        TraceAssertion._eval_report = report
        eval_matrix.record_eval_rows_from_report(report)

        s = report["summary"]
        if s["errors"] > 0 or s["failed"] > fail_threshold:
            failures = [r for r in report["scenarios"] if r["status"] != "pass"]
            lines = "\n".join(f"  {r['status']:7} {r['fact_id']}  exp={r['expected']} act={r['actual']}"
                              for r in failures)
            raise AssertionError(message or
                f"Filtered eval failed: {s['failed']} failed, {s['errors']} errors "
                f"(of {s['total']}).\n{lines}")
        return self

    def get_eval_report(self) -> Optional[dict]:
        """The uniform eval report stashed by check_eval (filter mode = N facts; span mode = 1 fact)."""
        return getattr(TraceAssertion, "_eval_report", None)

    def get_eval_failures(self) -> list:
        """Extract failures from the class-scoped eval report."""
        report = getattr(TraceAssertion, "_eval_report", None) or {"scenarios": []}
        return [r for r in report["scenarios"] if r["status"] != "pass"]

    def write_eval_report(self, path: str) -> None:
        """Write the class-scoped eval report to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(getattr(TraceAssertion, "_eval_report", {}) or {}, f, indent=2)

    @collect_assertions
    def under_token_limit(self, token_limit:int, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that all spans have total tokens under the given limit."""
        self.validator.check_total_token_limits(token_limit, filtered_spans=self._filtered_spans, custom_message=message)
        return self

    @collect_assertions
    def under_duration(self, duration_limit: float, units: str = "seconds", span_type:Optional[str] = "workflow", message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the workflow span's duration is under the given limit."""
        self.validator.check_duration_limits(duration_limit, filtered_spans=self._filtered_spans, units=units, span_type=span_type, custom_message=message)
        return self

    def load_spans(self, spans:list[Span]) -> None:
        """Load spans into the validator's memory exporter for assertions."""
        self.validator.add_remote_spans(spans)

    def _verify_top_level_io(self, expected, *, field:str, comparer:BaseComparer,
                             positive_test:bool, message:Optional[str]) -> None:
        """Check a test case's own `field` against the spans currently in scope.

        A list means every entry must hold -- they are separate expectations
        about one output, not alternatives, so each is checked on its own and
        all the failures are reported together.
        """
        expectations = [expected] if isinstance(expected, str) else list(expected)
        failures = []
        for value in expectations:
            if not value:
                continue
            io_kwargs = {"expected_inputs": [value], "expected_outputs": []} \
                if field == "input" else \
                {"expected_inputs": [], "expected_outputs": [value]}
            matched = self.validator._check_input_output(
                self._filtered_spans, comparer=comparer, eval=self._eval,
                positive_test=positive_test, **io_kwargs)
            if positive_test and not matched:
                failures.append(f"no {field} matching {value!r}")
            elif not positive_test and matched:
                failures.append(
                    f"{field} matching {value!r}, which was not expected")

        if failures:
            raise AssertionError(message or (
                f"{len(failures)} of {len(expectations)} {field} checks failed:"
                + "".join(f"{os.linesep}  - {failure}" for failure in failures)))

    def _verify_io_testcase(self, testcase, *, field:str, comparer:BaseComparer,
                            positive_test:bool, message:Optional[str]) -> None:
        """Check each test-case agent's own `field` against that agent's spans.

        Walks EVERY entry in tc.agents, duplicates included, and looks its spans
        up by name -- so two entries for one agent run two checks against one
        span list. That is what a single queue per agent name means: the
        expectations are per entry, the spans are per name.

        An entry that sets no value for `field` -- unset or empty -- states no
        expectation, so it is skipped. A call where *no* entry sets one asserts
        nothing and passes quietly: a test case describes what it knows, and a
        check it says nothing about is not a failure of that check.

        Uses validator._check_input_output, which returns the matching spans
        rather than raising, so failures can be accumulated. All of them are
        raised together because record_assertion keeps only the first per chain.
        """
        testcase = resolve_testcase(testcase)
        if self._entity_spans is None:
            # No selector ran, so there are no per-entity expectations. A
            # top-level `output` is the plain end-to-end check -- assert it
            # against whatever spans are in scope and stop there.
            top_level = getattr(testcase, field, None)
            if top_level:
                self._verify_top_level_io(
                    top_level, field=field, comparer=comparer,
                    positive_test=positive_test, message=message)
                return
            raise ValueError(
                "no entities selected; chain called_agent(testcase=...) or "
                "called_tool(testcase=...) before an input/output check that "
                f"takes a testcase, or give the testcase a top-level '{field}'.")

        # The selector that built the map decides which list describes it.
        kind = self._testcase_selector or "agent"
        entities = (testcase.tools if kind == "tool" else testcase.agents) or []

        checked, failures = 0, []
        for agent in entities:
            expected = getattr(agent, field)
            if not expected:
                # Unset or empty: the test case states no expectation for this
                # agent's input/output, so there is nothing to check. Covers ""
                # as well as None -- an empty string asserts nothing either.
                continue
            spans = self._entity_span_list(agent)
            if spans is None:
                # called_agent already reported this agent as not called; saying
                # so again in every I/O check would bury the real failures.
                continue
            checked += 1
            if field == "input":
                io_kwargs = {"expected_inputs": [expected], "expected_outputs": []}
            else:
                io_kwargs = {"expected_inputs": [], "expected_outputs": [expected]}
            matched = self.validator._check_input_output(
                spans, comparer=comparer, eval=self._eval,
                positive_test=positive_test, **io_kwargs)
            if positive_test and not matched:
                failures.append(
                    f"{kind} '{agent.name}' has no {field} matching {expected!r}")
            elif not positive_test and matched:
                failures.append(
                    f"{kind} '{agent.name}' has a {field} matching {expected!r}, "
                    "which was not expected")

        if failures:
            raise AssertionError(message or (
                f"{len(failures)} of {checked} {kind} {field} checks failed:"
                + "".join(f"{os.linesep}  - {failure}" for failure in failures)))

    def _verify_input_output(self, spans:list[Span], expected_inputs:Optional[list[str]], expected_outputs:Optional[list[str]],
                        comparer:BaseComparer, eval:Optional[Evaluation], positive_test:Optional[bool]=True,
                        tool_name:Optional[str]=None, agent_name:Optional[str]=None, custom_message:Optional[str]=None) -> None:
        filtered_spans: list[Span] = self.validator._check_input_output(spans, expected_inputs, expected_outputs,
                                                            comparer, eval, positive_test, tool_name, agent_name)
        if positive_test == True:
            self._filtered_spans = filtered_spans

        TraceAssertion._assert_on_spans(filtered_spans, "No matching operation found", positive_test, expected_inputs, expected_outputs, custom_message)

    def _verify_scope(self, spans:list[Span], scope_name:str, expected_values:Optional[list[str]],
                      comparer:BaseComparer, positive_test:Optional[bool]=True, custom_message:Optional[str]=None) -> None:
        """Verify that spans have the specified scope with expected value(s)."""
        filtered_spans: list[Span] = self.validator._check_scope(spans, scope_name, expected_values, comparer, positive_test)
        
        if positive_test == True:
            self._filtered_spans = filtered_spans

        if expected_values and len(expected_values) > 0:
            scope_description = f"scope '{scope_name}' with value(s) {expected_values}"
        else:
            scope_description = f"scope '{scope_name}'"
        
        if positive_test:
            assertion_message = f"No spans found with {scope_description}"
        else:
            assertion_message = f"Found spans with {scope_description}"
        
        TraceAssertion._assert_on_spans(filtered_spans, assertion_message, positive_test, custom_message=custom_message)

    @staticmethod
    def _assert_on_spans(spans:list[Span], assertion_message:str, positive_test:bool = True,
                    expected_inputs:Optional[list[str]] = None, expected_outputs:Optional[list[str]] = None,
                    custom_message:Optional[str] = None) -> None:
        if custom_message:
            # Use custom message if provided
            if positive_test == True and (not spans or len(spans) == 0):
                raise AssertionError(custom_message)
            if positive_test == False and spans and len(spans) > 0:
                raise AssertionError(custom_message)
        else:
            # Use default message
            if positive_test == True and (not spans or len(spans) == 0):
                if expected_inputs:
                    assertion_message += f" with expected inputs: {expected_inputs}."
                if expected_outputs:
                    assertion_message += f" with expected outputs: {expected_outputs}."
                raise AssertionError(assertion_message)
            if positive_test == False and spans and len(spans) > 0:
                if expected_inputs:
                    assertion_message += f" with unexpected inputs: {expected_inputs}."
                if expected_outputs:
                    assertion_message += f" with unexpected outputs: {expected_outputs}."
                raise AssertionError(assertion_message)
