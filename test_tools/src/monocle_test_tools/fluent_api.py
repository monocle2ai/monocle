from typing import Optional, Union

from .comparer.comparer_manager import get_comparer
from .comparer.base_comparer import BaseComparer
from .comparer.default_comparer import DefaultComparer
from .comparer.token_match_comparer import TokenMatchComparer
from .evals.eval_manager import get_evaluator
from .evals.base_eval import BaseEval
from .validator import MonocleValidator
from opentelemetry.sdk.trace import Span

class TraceAssertion(MonocleValidator):
    
    """Fluent API for asserting properties on Monocle traces."""
    _filtered_spans:Span = []
    _eval:Optional[Union[str, BaseEval]]  = None
    _comparer: Union[str, BaseComparer] = DefaultComparer()

    @staticmethod
    def get_trace_asserter():
        traceAssertion = TraceAssertion()
        traceAssertion.cleanup()
        return traceAssertion

    def with_evaluation(self, eval:Union[str, BaseEval]) -> 'TraceAssertion':
        self._eval = get_evaluator(eval)
        return self

    def with_comparer(self, comparer:Union[str, BaseComparer]) -> 'TraceAssertion':
        self._comparer = get_comparer(comparer)
        return self

    def with_expecting_errors(self, expect_errors:bool) -> 'TraceAssertion':
        self.expect_errors = expect_errors
        return self

    def with_expecting_warnings(self, expect_warnings:bool) -> 'TraceAssertion':
        self.expect_warnings = expect_warnings
        return self

    def called_tool(self, tool_name:str, agent_name:Optional[str] = None) -> 'TraceAssertion':
        self._filtered_spans = self._get_tool_invocation_spans(tool_name, agent_name)
        if agent_name:
            self._assert_on_spans(f"Tool '{tool_name}' was not called by agent '{agent_name}'")
        else:
            self._assert_on_spans(f"Tool '{tool_name}' was not called")
        return self

    def called_agent(self, agent_name:str) -> 'TraceAssertion':
        self._filtered_spans = self._get_agent_invocation_spans(agent_name)
        self._assert_on_spans(f"Agent '{agent_name}' was not called")
        return self

    def has_input(self, expected_input:str) -> 'TraceAssertion':
        self._filtered_spans = self._check_input_output(self._filtered_spans, expected_input=expected_input,
                                    expected_output=None, comparer=self._comparer, eval=self._eval)
        return self

    def does_not_have_input(self, unexpected_input:str) -> 'TraceAssertion':
        self._filtered_spans = self._check_input_output(self._filtered_spans, expected_input=unexpected_input,
                                    expected_output=None, comparer=self._comparer, eval=self._eval, positive_test=False)
        return self

    def contains_input(self, expected_input_substring:str) -> 'TraceAssertion':
        self._filtered_spans = self._check_input_output(self._filtered_spans, expected_input=expected_input_substring,
                                    expected_output=None, comparer=TokenMatchComparer(), eval=self._eval)
        return self

    def does_not_contain_input(self, unexpected_input_substring:str) -> 'TraceAssertion':
        self._filtered_spans = self._check_input_output(self._filtered_spans, expected_input=unexpected_input_substring,
                                    expected_output=None, comparer=TokenMatchComparer(), eval=self._eval, positive_test=False)
        return self

    def has_output(self, expected_output:str) -> 'TraceAssertion':
        self._filtered_spans = self._check_input_output(self._filtered_spans, expected_input=None, expected_output=expected_output,
                                 comparer=self._comparer, eval=self._eval)
        return self

    def contains_output(self, expected_output_substring:str) -> 'TraceAssertion':
        self._filtered_spans = self._check_input_output(self._filtered_spans, expected_input=None,
                                expected_output=expected_output_substring, comparer=TokenMatchComparer(), eval=self._eval)
        return self

    def does_not_contain_output(self, unexpected_output_substring:str) -> 'TraceAssertion':
        self._filtered_spans = self._check_input_output(self._filtered_spans, expected_input=None,
                                expected_output=unexpected_output_substring, comparer=TokenMatchComparer(), eval=self._eval,
                                positive_test=False)
        return self

    def does_not_have_output(self, unexpected_output:str) -> 'TraceAssertion':
        self._filtered_spans = self._check_input_output(self._filtered_spans, expected_input=None, expected_output=unexpected_output,
                                 comparer=self._comparer, eval=self._eval, positive_test=False)
        return self

    def cleanup(self) -> None:
        self.memory_exporter.clear()
        self._filtered_spans = []

    def _assert_on_spans(self, assertion_message:str) -> None:
        if not self._filtered_spans or len(self._filtered_spans) == 0:
            raise AssertionError(assertion_message)
