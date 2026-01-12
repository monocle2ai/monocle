from functools import wraps
import inspect
import os
from typing import Optional, Union
from monocle_test_tools.schema import Evaluation
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
                            is_assertion_failed=asserter.is_assertion_failed)
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

    def __init__(self, filtered_spans:Optional[Span] = [], fluent_chain:list[str] = []
                ,is_assertion_failed:bool = False):
        self.validator = MonocleValidator()
        self._filtered_spans = filtered_spans
        self.fluent_chain = fluent_chain
        self.is_assertion_failed = is_assertion_failed

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
        self.validator.cleanup()
        self._filtered_spans = []
        TraceAssertion._assertion_errors = []

    def run_agent(self, agent, agent_type:str, *args, **kwargs) -> any:
        """Run the given agent with provided args and kwargs."""
        return self.validator.run_agent(agent, agent_type, *args, **kwargs)

    async def run_agent_async(self, agent, agent_type:str, *args, **kwargs) -> any:
        """Run the given async agent with provided args and kwargs."""
        return await self.validator.run_agent_async(agent, agent_type, *args, **kwargs)

    def with_evaluation(self, eval:Union[str, BaseEval]) -> 'TraceAssertion':
        """Set the evaluation method for input/output comparisons."""
        self._eval = get_evaluator(eval)
        return self

    def with_comparer(self, comparer:Union[str, BaseComparer]) -> 'TraceAssertion':
        """Set the comparer for input/output comparisons."""
        self._comparer = get_comparer(comparer)
        return self

    @collect_assertions
    def called_tool(self, tool_name:str, agent_name:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the given tool was called, optionally by a specific agent."""
        self._filtered_spans = self.validator._get_tool_invocation_spans(tool_name, agent_name)
        if agent_name:
            TraceAssertion._assert_on_spans(self._filtered_spans, f"Tool '{tool_name}' was not called by agent '{agent_name}'")
        else:
            TraceAssertion._assert_on_spans(self._filtered_spans, f"Tool '{tool_name}' was not called")
        return self

    @collect_assertions
    def does_not_call_tool(self, tool_names:str, agent_name:Optional[str] = None) -> 'TraceAssertion':
        """Assert that the given tool was not called, optionally by a specific agent."""
        _filtered_spans = self.validator._get_tool_invocation_spans(tool_names, agent_name)
        if agent_name:
            TraceAssertion._assert_on_spans(_filtered_spans, f"Tool '{tool_names}' was called by agent '{agent_name}'", positive_test=False)
        else:
            TraceAssertion._assert_on_spans(_filtered_spans, f"Tool '{tool_names}' was called", positive_test=False)
        return self

    @collect_assertions
    def called_agent(self, agent_name:str) -> 'TraceAssertion':
        """Assert that the given agent was called."""
        self._filtered_spans = self.validator._get_agent_invocation_spans(agent_name)
        TraceAssertion._assert_on_spans(self._filtered_spans, f"Agent '{agent_name}' was not called")
        return self

    @collect_assertions
    def does_not_call_agent(self, agent_name:str) -> 'TraceAssertion':
        """Assert that the given agent was not called."""
        _filtered_spans = self.validator._get_agent_invocation_spans(agent_name)
        TraceAssertion._assert_on_spans(_filtered_spans, f"Agent '{agent_name}' was called", positive_test=False)
        return self

    @collect_assertions
    def has_input(self, expected_input:str) -> 'TraceAssertion':
        """Assert that the input matches the expected input."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[expected_input],
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval)
        return self

    @collect_assertions
    def has_any_input(self, *expected_inputs:str) -> 'TraceAssertion':
        """Assert that any of the expected inputs match."""
        if not expected_inputs:
            raise ValueError("At least one expected_input is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(expected_inputs),
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval)
        return self

    @collect_assertions
    def does_not_have_input(self, unexpected_input:str) -> 'TraceAssertion':
        """Assert that the input does not match the unexpected input."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[unexpected_input],
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, positive_test=False)
        return self

    @collect_assertions
    def does_not_have_any_input(self, *unexpected_inputs:str) -> 'TraceAssertion':
        """Assert that none of the unexpected inputs match."""
        if not unexpected_inputs:
            raise ValueError("At least one unexpected_input is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(unexpected_inputs),
                                    expected_outputs=[], comparer=self._comparer, eval=self._eval, positive_test=False)
        return self

    @collect_assertions
    def contains_input(self, expected_input_substring:str) -> 'TraceAssertion':
        """Assert that the input contains the expected substring"""
        self._verify_input_output(self._filtered_spans, expected_inputs=[expected_input_substring],
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval)
        return self

    @collect_assertions
    def contains_any_input(self, *expected_input_substrings:str) -> 'TraceAssertion':
        """Assert that any input contains the expected substring"""
        if not expected_input_substrings:
            raise ValueError("At least one expected_input_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(expected_input_substrings),
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval)
        return self

    @collect_assertions
    def does_not_contain_input(self, unexpected_input_substring:str) -> 'TraceAssertion':
        """Assert that the input does not contain the given substring"""
        self._verify_input_output(self._filtered_spans, expected_inputs=[unexpected_input_substring],
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, positive_test=False)
        return self

    @collect_assertions
    def does_not_contain_any_input(self, *unexpected_input_substrings:str) -> 'TraceAssertion':
        """Assert that no input contains the given substrings"""
        if not unexpected_input_substrings:
            raise ValueError("At least one unexpected_input_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=list(unexpected_input_substrings),
                                    expected_outputs=[], comparer=TokenMatchComparer(), eval=self._eval, positive_test=False)
        return self

    @collect_assertions
    def has_output(self, expected_output:str) -> 'TraceAssertion':
        """Assert that the output matches the expected output."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[], expected_outputs=[expected_output],
                                 comparer=self._comparer, eval=self._eval)
        return self

    @collect_assertions
    def has_any_output(self, *expected_outputs:str) -> 'TraceAssertion':
        """Assert that the output matches any of the expected outputs."""
        if not expected_outputs:
            raise ValueError("At least one expected_output is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                    expected_outputs=list(expected_outputs), comparer=self._comparer, eval=self._eval)
        return self

    @collect_assertions
    def does_not_have_output(self, unexpected_output:str) -> 'TraceAssertion':
        """Assert that the output does not have the given output."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[] , expected_outputs=[unexpected_output],
                                 comparer=self._comparer, eval=self._eval, positive_test=False)
        return self

    @collect_assertions
    def does_not_have_any_output(self, *unexpected_outputs:str) -> 'TraceAssertion':
        """Assert that the output does not have any of the given outputs."""
        if not unexpected_outputs:
            raise ValueError("At least one unexpected_output is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                 expected_outputs=list(unexpected_outputs), comparer=self._comparer, eval=self._eval, positive_test=False)
        return self

    @collect_assertions
    def contains_output(self, expected_output_substring:str) -> 'TraceAssertion':
        """Assert that the output contains the expected substring."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=[expected_output_substring], comparer=TokenMatchComparer(), eval=self._eval)
        return self

    @collect_assertions
    def contains_any_output(self, *expected_output_substrings:str) -> 'TraceAssertion':
        """Assert that any output contains the expected substring."""
        if not expected_output_substrings:
            raise ValueError("At least one expected_output_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=list(expected_output_substrings), comparer=TokenMatchComparer(), eval=self._eval)
        return self

    @collect_assertions
    def does_not_contain_output(self, unexpected_output_substring:str) -> 'TraceAssertion':
        """Assert that the output does not contain the given substring."""
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=[unexpected_output_substring], comparer=TokenMatchComparer(), eval=self._eval,
                                positive_test=False)
        return self

    @collect_assertions
    def does_not_contain_any_output(self, *unexpected_output_substrings:str) -> 'TraceAssertion':
        """Assert that no output contains the given substrings."""
        if not unexpected_output_substrings:
            raise ValueError("At least one unexpected_output_substring is required")
        self._verify_input_output(self._filtered_spans, expected_inputs=[],
                                expected_outputs=list(unexpected_output_substrings), comparer=TokenMatchComparer(), eval=self._eval,
                                positive_test=False)
        return self

    @collect_assertions
    def under_token_limit(self, token_limit:int) -> 'TraceAssertion':
        """Assert that all spans have total tokens under the given limit."""
        self.validator._check_token_limit(self._filtered_spans, token_limit)
        return self

    def load_spans(self, spans:list[Span]) -> None:
        """Load spans into the validator's memory exporter for assertions."""
        self.validator.memory_exporter.export(spans)

    def _verify_input_output(self, spans:list[Span], expected_inputs:Optional[list[str]], expected_outputs:Optional[list[str]],
                        comparer:BaseComparer, eval:Optional[Evaluation], positive_test:Optional[bool]=True,
                        tool_name:Optional[str]=None, agent_name:Optional[str]=None) -> None:
        filtered_spans: list[Span] = self.validator._check_input_output(spans, expected_inputs, expected_outputs,
                                                            comparer, eval, positive_test, tool_name, agent_name)
        if positive_test == True:
            self._filtered_spans = filtered_spans
        TraceAssertion._assert_on_spans(filtered_spans, "No matching operation found with given input/output criteria.", positive_test)

    @staticmethod
    def _assert_on_spans(spans:list[Span], assertion_message:str, positive_test:bool = True) -> None:
        if positive_test == True and (not spans or len(spans) == 0):
            raise AssertionError(assertion_message)
        if positive_test == False and spans and len(spans) > 0:
            raise AssertionError(assertion_message)
