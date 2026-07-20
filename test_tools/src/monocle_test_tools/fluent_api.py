from functools import wraps
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union
from monocle_apptrace.instrumentation.common.method_wrappers import monocle_trace_method
from monocle_apptrace.instrumentation.common.utils import get_workflow_name
from monocle_test_tools import eval_matrix
from monocle_test_tools.schema import Evaluation
from monocle_test_tools.span_loader import JSONSpanLoader, OkahuSpanLoader
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
        asserter = TraceAssertion(filtered_spans=asserter._filtered_spans, fluent_chain=fluent_chain,
                            is_assertion_failed=asserter.is_assertion_failed, _eval=asserter._eval,
                            okahu_filter=getattr(asserter, "_okahu_filter", None))
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
    # Filter scope recorded by with_trace_source("okahu", start_time=..., end_time=...)
    # for eval-only filtered runs. Threaded through @collect_assertions like
    # _filtered_spans so the (decorated) check_eval can read it.
    _okahu_filter: Optional[dict] = None

    @staticmethod
    def get_trace_asserter():
        traceAssertion = TraceAssertion()
        traceAssertion.cleanup()
        return traceAssertion

    def __init__(self, filtered_spans:Optional[list[Span]] = None, fluent_chain:list[str] = []
                ,is_assertion_failed:bool = False, _eval:Optional[Union[str, BaseEval]] = None,
                okahu_filter:Optional[dict] = None) -> None:
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

    def run_agent(self, agent, agent_type:str, *args, **kwargs) -> any:
        """Run the given agent with provided args and kwargs."""
        return self.validator.run_agent(agent, agent_type, *args, mock_tools=self.mock_tools, **kwargs)

    async def run_agent_async(self, agent, agent_type:str, *args, session_id:str=None, **kwargs) -> any:
        """Run the given async agent with provided args and kwargs."""
        return await self.validator.run_agent_async(agent, agent_type, *args, session_id=session_id, mock_tools=self.mock_tools, **kwargs)

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

    def with_trace_source(self, source: str = "local", **kwargs) -> 'TraceAssertion':
        """Configure trace source for assertions.

        Args:
            source: Trace source type:
                - ``"local"`` (default) — Use traces from memory (current execution).
                - ``"file"`` — Load traces from local .monocle/*.json files.
                - ``"okahu"`` — Fetch traces from Okahu cloud.
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
                raise ValueError("Provide an 'id' or a time window (start_time/end_time), not both.")
            if has_window:
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
    def called_tool(self, tool_name:str, agent_name:Optional[str] = None, count:Optional[int] = None,
                    min_count:Optional[int] = None, max_count:Optional[int] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert tool invocation with optional agent filter and count constraints (count, min_count, max_count)."""
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
    def called_agent(self, agent_name:str, count:Optional[int] = None, min_count:Optional[int] = None, 
                     max_count:Optional[int] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert agent invocation with optional count constraints (count, min_count, max_count)."""
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
    def has_input(self, expected_input:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the input matches the expected input."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[expected_input],
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def has_any_input(self, *expected_inputs:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that any of the expected inputs match."""
        if not expected_inputs:
            raise ValueError("At least one expected_input is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(expected_inputs),
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_input(self, unexpected_input:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the input does not match the unexpected input."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[unexpected_input],
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_any_input(self, *unexpected_inputs:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that none of the unexpected inputs match."""
        if not unexpected_inputs:
            raise ValueError("At least one unexpected_input is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(unexpected_inputs),
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def contains_input(self, expected_input_substring:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the input contains the expected substring"""
        self._verify_input_output(self._filtered_spans, expected_inputs=[expected_input_substring],
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def contains_any_input(self, *expected_input_substrings:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that any input contains the expected substring"""
        if not expected_input_substrings:
            raise ValueError("At least one expected_input_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(expected_input_substrings),
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_input(self, unexpected_input_substring:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the input does not contain the given substring"""
        self._verify_input_output(self._filtered_spans, expected_inputs=[unexpected_input_substring],
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_any_input(self, *unexpected_input_substrings:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that no input contains the given substrings"""
        if not unexpected_input_substrings:
            raise ValueError("At least one unexpected_input_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(unexpected_input_substrings),
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def has_output(self, expected_output:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the output matches the expected output."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[], expected_outputs=[expected_output],
                                 comparer=self._comparer, eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def has_any_output(self, *expected_outputs:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the output matches any of the expected outputs."""
        if not expected_outputs:
            raise ValueError("At least one expected_output is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                    expected_outputs=list(expected_outputs), comparer=self._comparer, eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_output(self, unexpected_output:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the output does not have the given output."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[] , expected_outputs=[unexpected_output],
                                 comparer=self._comparer, eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_have_any_output(self, *unexpected_outputs:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the output does not have any of the given outputs."""
        if not unexpected_outputs:
            raise ValueError("At least one unexpected_output is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                 expected_outputs=list(unexpected_outputs), comparer=self._comparer, eval=self._eval, positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def contains_output(self, expected_output_substring:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the output contains the expected substring."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=[expected_output_substring], comparer=TokenMatchComparer(), eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def contains_any_output(self, *expected_output_substrings:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that any output contains the expected substring."""
        if not expected_output_substrings:
            raise ValueError("At least one expected_output_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=list(expected_output_substrings), comparer=TokenMatchComparer(), eval=self._eval, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_output(self, unexpected_output_substring:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the output does not contain the given substring."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=[unexpected_output_substring], comparer=TokenMatchComparer(), eval=self._eval,
                                positive_test=False, custom_message=message)
        return self

    @collect_assertions
    def does_not_contain_any_output(self, *unexpected_output_substrings:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that no output contains the given substrings."""
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

    @collect_assertions
    def check_eval(self, eval_name:Optional[str] = None, expected:Optional[Union[str, list[str]]] = None, not_expected:Optional[Union[str, list[str]]] = None, fact_name:Optional[str] = "traces", message:Optional[str] = None, template_path:Optional[str] = None) -> 'TraceAssertion':
        """Validate evaluation results for the current filtered spans.

        Provide exactly one of:
          - eval_name: name of a standard Okahu eval template (e.g. "hallucination")
          - template_path: filesystem path to a custom-template JSON file. The file
            is loaded and the parsed dict is sent to the eval service. Server-side
            validation errors (HTTP 400) surface as AssertionError with the prefix
            'Custom template validation failed: <reason>'.
        """
        if eval_name and template_path:
            raise ValueError("Provide either 'eval_name' or 'template_path', not both.")
        if not eval_name and not template_path:
            raise ValueError("Provide either 'eval_name' (for Okahu templates) or 'template_path' (for custom templates).")

        template = None
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

        eval_result, explanation = self._eval.evaluate(filtered_spans=self._filtered_spans, eval_name=eval_name, fact_name=fact_name, template=template)

        self._last_eval.update(
            label=eval_result,
            explanation=explanation,
            judge_output=getattr(self._eval, "last_judge_output", {}) or {},
            total_tokens=getattr(self._eval, "last_total_tokens", None),
        )

        if (positive and eval_result not in positive) or (negative and eval_result in negative):
            if message:
                raise AssertionError(message)
            elif positive and eval_result not in positive:
                raise AssertionError(f"Evaluation '{eval_name}' did not match expected result. Expected one of {positive}. Received '{eval_result}'. \n Explanation: {explanation}")
            else:
                raise AssertionError(f"Evaluation '{eval_name}' matched an unexpected result. Should not be any of {negative}. Received '{eval_result}'. \n Explanation: {explanation}")

        return self

    def with_filtered_source(self, source: str = "okahu", *, workflow_name,
                             start_time, end_time, fact_name: str = "traces") -> 'TraceAssertion':
        """Declare a filtered eval scope (async job over a workflow + time-window filter).

        Unlike with_trace_source (which imports named traces), this records the filter;
        the job is submitted by check_eval_filtered. `workflow_name` is a str or list.
        Not decorated with @collect_assertions: it records scope on the current asserter
        instance so the (also-undecorated) check_eval_filtered can read it.
        """
        if source != "okahu":
            raise ValueError("with_filtered_source currently supports only source='okahu'.")
        workflows = [workflow_name] if isinstance(workflow_name, str) else list(workflow_name)
        self._filtered_scope = {"workflows": workflows, "start_time": start_time,
                                "end_time": end_time, "fact_name": fact_name}
        return self

    def check_eval_filtered(self, eval_name: Optional[str] = None,
                            expected: Optional[Union[str, list[str]]] = None,
                            not_expected: Optional[Union[str, list[str]]] = None, *,
                            template_path: Optional[str] = None,
                            template: Optional[dict] = None,
                            fail_threshold: int = 0, min_facts: int = 1,
                            message: Optional[str] = None) -> 'TraceAssertion':
        """Run a filtered eval over the declared filtered source and assert labels.

        expected (str|list, any-of) and/or not_expected (str|list) are applied to EVERY
        fact the job discovers (min_facts guards a vacuous pass). Exactly one template
        selector: eval_name XOR template_path XOR template. The whole test fails only
        after the job reaches a terminal state, raising a single assertion with a
        per-fact failure table.
        """
        from monocle_test_tools.evals.okahu_filtered_eval import OkahuFilteredEval
        scope = getattr(self, "_filtered_scope", None)
        if scope is None:
            raise AssertionError("No filtered source. Call with_filtered_source before check_eval_filtered.")
        selectors = [s for s in (eval_name, template_path, template) if s]
        if len(selectors) != 1:
            raise ValueError("Provide exactly one of 'eval_name', 'template_path', or 'template'.")

        # Blanket-expectation shape.
        if isinstance(expected, dict):
            raise ValueError("check_eval_filtered takes 'expected' as a str/list, not a dict.")
        if expected is None and not_expected is None:
            raise ValueError("check_eval_filtered requires 'expected' and/or 'not_expected'.")
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
            start_time=scope["start_time"], end_time=scope["end_time"], min_facts=min_facts)
        self._filtered_report = report

        # Feed the shared PR #722 eval-matrix recorder (Phase 4). Self-skips if disabled.
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

    def get_filtered_eval_report(self) -> dict:
        return getattr(self, "_filtered_report", None)

    def get_filtered_eval_failures(self) -> list:
        report = getattr(self, "_filtered_report", None) or {"scenarios": []}
        return [r for r in report["scenarios"] if r["status"] != "pass"]

    def write_filtered_eval_report(self, path: str) -> None:
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(getattr(self, "_filtered_report", {}), f, indent=2)

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
