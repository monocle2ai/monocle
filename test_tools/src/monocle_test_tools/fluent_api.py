from functools import wraps
import inspect
import os
from typing import Optional, Union
from monocle_test_tools.schema import Evaluation
from monocle_test_tools.span_loader import JSONSpanLoader
from .comparer.comparer_manager import get_comparer
from .comparer.base_comparer import BaseComparer
from .comparer.default_comparer import DefaultComparer
from .comparer.token_match_comparer import TokenMatchComparer
from .evals.eval_manager import get_evaluator
from .evals.base_eval import BaseEval
from .validator import MonocleValidator
from .trace_utils import get_function_signature, get_caller_file_line
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
                            is_assertion_failed=asserter.is_assertion_failed, _eval=asserter._eval)
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

    @staticmethod
    def get_trace_asserter():
        traceAssertion = TraceAssertion()
        traceAssertion.cleanup()
        return traceAssertion

    def __init__(self, filtered_spans:Optional[list[Span]] = None, fluent_chain:list[str] = []
                ,is_assertion_failed:bool = False, _eval:Optional[Union[str, BaseEval]] = None) -> None:
        self._eval:Union[str, BaseEval]  = _eval
        self.validator = MonocleValidator()
        if filtered_spans is None:
            if self.validator.spans is not None and len(self.validator.spans) > 0:
                filtered_spans = self.validator.spans
        self._filtered_spans = filtered_spans
        self.fluent_chain = fluent_chain
        self.is_assertion_failed = is_assertion_failed
        self._skip_export = False
        
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
        TraceAssertion._assertion_errors = []

    def run_agent(self, agent, agent_type:str, *args, **kwargs) -> any:
        """Run the given agent with provided args and kwargs."""
        return self.validator.run_agent(agent, agent_type, *args, **kwargs)

    async def run_agent_async(self, agent, agent_type:str, *args, session_id:str=None, **kwargs) -> any:
        """Run the given async agent with provided args and kwargs."""
        return await self.validator.run_agent_async(agent, agent_type, *args, session_id=session_id, **kwargs)

    def with_evaluation(self, eval:Union[str, BaseEval], eval_options:Optional[dict] = None) -> 'TraceAssertion':
        """Set the evaluation method for input/output comparisons."""
        self._eval = get_evaluator(eval, eval_options)
        return self

    def with_comparer(self, comparer:Union[str, BaseComparer]) -> 'TraceAssertion':
        """Set the comparer for input/output comparisons."""
        self._comparer = get_comparer(comparer)
        return self

    @collect_assertions
    def called_tool(self, tool_name:str, agent_name:Optional[str] = None, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the given tool was called, optionally by a specific agent."""
        self._filtered_spans = self.validator._get_tool_invocation_spans(tool_name, agent_name, filtered_spans=self._filtered_spans)
        if agent_name:
            TraceAssertion._assert_on_spans(self._filtered_spans, f"Tool '{tool_name}' was not called by agent '{agent_name}'", custom_message=message)
        else:
            TraceAssertion._assert_on_spans(self._filtered_spans, f"Tool '{tool_name}' was not called", custom_message=message)
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
    def called_agent(self, agent_name:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the given agent was called."""
        self._filtered_spans = self.validator._get_agent_invocation_spans(agent_name, filtered_spans=self._filtered_spans)
        TraceAssertion._assert_on_spans(self._filtered_spans, f"Agent '{agent_name}' was not called", custom_message=message)
        return self

    @collect_assertions
    def does_not_call_agent(self, agent_name:str, message:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the given agent was not called."""
        _filtered_spans = self.validator._get_agent_invocation_spans(agent_name, filtered_spans=self._filtered_spans)
        TraceAssertion._assert_on_spans(_filtered_spans, f"Agent '{agent_name}' was called", positive_test=False, custom_message=message)
        return self

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
    def check_eval(self, eval_name:str, expected:Optional[Union[str, list[str]]] = None, not_expected:Optional[Union[str, list[str]]] = None, fact_name:Optional[str] = "traces", message:Optional[str] = None) -> 'TraceAssertion':
        """Validate evaluation results for the current filtered spans."""
        #verify expected and not_expected aren't empty
        if expected is None and not_expected is None:
            raise ValueError("At least one of 'expected' or 'not_expected' must be provided")
        # Convert strings to lists for uniform processing
        positive = [expected] if isinstance(expected, str) else expected if expected is not None else []
        negative = [not_expected] if isinstance(not_expected, str) else not_expected if not_expected is not None else []
        
        # Check for overlapping instances in expected and not_expected
        if negative:
            overlap = set(positive) & set(negative)
            if overlap:
                raise ValueError(f"Overlapping evaluation results found in 'expected' and 'not_expected': {overlap}. Please ensure they are mutually exclusive.")
        
        if self._eval is None:
            raise AssertionError(message if message else "No evaluator configured. Call with_evaluation before check_eval.")
        if not self._filtered_spans:
            raise AssertionError(message if message else "No spans available for evaluation. Chain a span selector before check_eval.")
        eval_result = self._eval.evaluate(filtered_spans=self._filtered_spans, eval_name=eval_name, fact_name=fact_name)        
        
        # Check expectations
        if (positive and eval_result not in positive) or (negative and eval_result in negative):
            if message:
                raise AssertionError(message)
            elif positive and eval_result not in positive:
                raise AssertionError(f"Evaluation '{eval_name}' did not match expected result. Expected one of {positive}. Received '{eval_result}'.")
            else:
                raise AssertionError(f"Evaluation '{eval_name}' matched an unexpected result. Should not be any of {negative}. Received '{eval_result}'.")
        
        return self

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
        self.validator.memory_exporter.export(spans)

    def import_traces(self, trace_source:str = "file", trace_id:str = None,
                      trace_file:str = None, trace_dir:str = None) -> 'TraceAssertion':
        """Import traces from a file source for assertion.

        Loads previously exported trace spans into the asserter's memory
        so assertions can be run against them without re-running the agent.

        Args:
            trace_source: Source type, currently only "file" is supported.
            trace_id: The trace ID (hex string) to locate the trace file.
            trace_file: Direct path to a trace JSON file (overrides trace_id lookup).
            trace_dir: Directory to search for trace files. Defaults to .monocle/test_traces.

        Returns:
            self for fluent chaining.

        Raises:
            ValueError: If trace_source is not supported or required params are missing.
            FileNotFoundError: If no trace file is found for the given trace_id.
        """
        if trace_source != "file":
            raise ValueError(f"Unsupported trace_source: '{trace_source}'. Currently only 'file' is supported.")

        if trace_file is None and trace_id is None:
            raise ValueError("Either 'trace_id' or 'trace_file' must be provided.")

        if trace_file is None:
            trace_file = JSONSpanLoader.find_trace_file(trace_id, trace_dir)
            if trace_file is None:
                search_dir = trace_dir or os.path.join(".", ".monocle", "test_traces")
                raise FileNotFoundError(
                    f"No trace file found for trace_id '{trace_id}' in '{search_dir}'")

        spans = JSONSpanLoader.from_json(trace_file)
        self.load_spans(spans)
        # Refresh filtered spans from the newly loaded data
        self._filtered_spans = self.validator.spans
        # Skip re-exporting imported traces to avoid duplicate trace files
        self._skip_export = True
        return self

    def _verify_input_output(self, spans:list[Span], expected_inputs:Optional[list[str]], expected_outputs:Optional[list[str]],
                        comparer:BaseComparer, eval:Optional[Evaluation], positive_test:Optional[bool]=True,
                        tool_name:Optional[str]=None, agent_name:Optional[str]=None, custom_message:Optional[str]=None) -> None:
        filtered_spans: list[Span] = self.validator._check_input_output(spans, expected_inputs, expected_outputs,
                                                            comparer, eval, positive_test, tool_name, agent_name)
        if positive_test == True:
            self._filtered_spans = filtered_spans

        TraceAssertion._assert_on_spans(filtered_spans, "No matching operation found", positive_test, expected_inputs, expected_outputs, custom_message)

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
