import os
from functools import wraps
import inspect
import uuid
import jsonschema, json
from typing import Optional, Union
from opentelemetry.sdk.trace import Span, StatusCode
from opentelemetry.sdk.trace.export import SpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.context import set_value, Context
import pytest
#from sqlalchemy import func
from monocle_apptrace.exporters.file_exporter import FileSpanExporter, DEFAULT_TRACE_FOLDER
from monocle_apptrace import start_scopes, stop_scope
from contextlib import contextmanager, asynccontextmanager
import logging
from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter
from monocle_apptrace.instrumentation.common.instrumentor import MonocleInstrumentor, setup_monocle_telemetry, reset_span_processors, get_monocle_instrumentor
from pydantic import BaseModel, ValidationError
from monocle_test_tools.gitutils import get_git_context
from monocle_test_tools.schema import SpanType, TestSpan, TestCase, Evaluation, EvalInputs, MockTool
from monocle_test_tools.constants import TEST_SCOPE_NAME
from monocle_test_tools.comparer.base_comparer import BaseComparer
from monocle_test_tools.runner.runner import get_agent_runner
from monocle_test_tools import trace_utils
from monocle_apptrace.instrumentation.metamodel.adk.methods import ADK_METHODS
from monocle_apptrace.instrumentation.metamodel.adk.entities.tool import TOOL as ADK_TOOL
from monocle_apptrace.instrumentation.metamodel.langgraph.methods import LANGGRAPH_METHODS
from monocle_apptrace.instrumentation.metamodel.langgraph.entities.inference import TOOLS as LANGGRAPH_TOOL
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SKIP_EXECUTIONS

logger = logging.getLogger(__name__)

class MonocleValidator:
    _spans:Span = []
    memory_exporter:InMemorySpanExporter = None
    file_exporter:FileSpanExporter = None
    trace_id = None
    instrumentor: MonocleInstrumentor = None
    _instance = None
    _initialized = False
    exporters:list[SpanExporter] = []
    export_failed_tests_only: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, exporter_list:Optional[str] = None, export_failed_tests_only:Optional[bool] = None):
        if MonocleValidator._initialized:
            if exporter_list is not None:
                raise ValueError("Exporter list can only be set during the first initialization of MonocleValidator.")
            return
        test_trace_path:str = os.path.join(".", DEFAULT_TRACE_FOLDER, "test_traces")
        os.environ["MONOCLE_TRACE_OUTPUT_PATH"] = test_trace_path
        if exporter_list is None:
            exporter_list = os.getenv("MONOCLE_EXPORTER", "file")
        self.exporters = get_monocle_exporter(exporter_list)
        self.memory_exporter = InMemorySpanExporter()
        if export_failed_tests_only is None:
            export_failed_tests_only = os.getenv("MONOCLE_EXPORT_FAILED_TESTS_ONLY", "false").lower() == "true"
        self.export_failed_tests_only = export_failed_tests_only
        if get_monocle_instrumentor() is None:
            self.instrumentor = setup_monocle_telemetry(workflow_name="monocle_validator",
                                        span_processors=[SimpleSpanProcessor(self.memory_exporter)])
        else:
            self.instrumentor = get_monocle_instrumentor()
            reset_span_processors([SimpleSpanProcessor(self.memory_exporter)])
        MonocleValidator._initialized = True

    def __del__(self):
        for exporter in self.exporters:
            if hasattr(exporter, "shutdown"):
                exporter.shutdown()

    def cleanup(self):
        """Cleanup the validator state for a fresh test run."""
        self._spans = []
        if self.memory_exporter is not None:
            self.memory_exporter.clear()
        if self.file_exporter is not None:
            self.file_exporter.force_flush()
        self.trace_id = None

    @property
    def spans(self):
        if len(self._spans) == 0 and self.memory_exporter is not None:
            self._spans = self.memory_exporter.get_finished_spans()
        return self._spans

    def flush_to_exporters(self, test_name:str, test_failed:bool):
        """Flush the current spans and prepare for validation."""
        if self.export_failed_tests_only and not test_failed:
            return
        span:Span = None
        for exporter in self.exporters:
            for span in self.memory_exporter.get_finished_spans():
                if test_failed:
                    span._attributes["test.status"] = "failed"
                else:
                    span._attributes["test.status"] = "passed"
                exporter.export([span])
            if hasattr(exporter, "force_flush"):
                exporter.force_flush()

    def pre_test_run_setup(self, test_case_name:str, mock_tools: Optional[list[MockTool]] = None) -> None:
        """
        Prepares the validator for a new test run by clearing existing spans.
        """
        context:Context = None
        if mock_tools is not None:
            context = self._set_wrapper_methods(mock_tools)
        test_scope = {TEST_SCOPE_NAME: test_case_name}
        git_scopes = get_git_context()
        all_scopes = {**test_scope, **git_scopes}
        token = start_scopes(all_scopes, context)
        return token

    def post_test_cleanup(self, token:object, test_name:str, test_failed:bool):
        try:
            self.flush_to_exporters(test_name, test_failed)
        finally:
            self.cleanup()
            if token is not None:
                stop_scope(token)

    @contextmanager
    def monocle_exporter_wrapper(self, test_case: TestCase, request:pytest.FixtureRequest):
        test_case_name = request.node.name if request is not None else (test_case.test_case_name if test_case is not None else "monocle_test")
        token = self.pre_test_run_setup(test_case_name, test_case.mock_tools)
        prior_test_failed_count = request.session.testsfailed
        try:
            yield
        finally:
            try:
                self.validate(test_case)
            finally:
                self.post_test_cleanup(token, request.node.name, request.session.testsfailed > prior_test_failed_count)
                self._spans = []
                if token is not None:
                    stop_scope(token)

    @staticmethod
    def test_id_generator(val):
        return f"{val.test_name}_{uuid.uuid4().hex[:8]}"

    def monocle_testcase(self, test_cases_array: list[Union[TestCase, dict]]):
        test_cases: list[TestCase] = []
        for tc in test_cases_array:
            if isinstance(tc, dict):
                test_cases.append(TestCase.model_validate(tc))
            else:
                test_cases.append(tc)
        """Decorator to mark a test function as a monocle test case."""
        def decorator(func):
            if inspect.iscoroutinefunction(func):
                @pytest.mark.asyncio
                @pytest.mark.parametrize("test_case", test_cases)
                async def wrapper(test_case, request, *args, **kwargs):
                    with self.monocle_exporter_wrapper(test_case, request):
                        return await func(test_case, *args, **kwargs)
            else:
                @pytest.mark.parametrize("test_case", test_cases)
                def wrapper(test_case, request, *args, **kwargs):
                    with self.monocle_exporter_wrapper(test_case, request):
                        return func(test_case, *args, **kwargs)
            return wrapper
        return decorator

    async def test_workflow_async(self, workflow_func, test_case:Union[TestCase, dict]):
        """Run the workflow function with the test case input and validate the output.
        Args:
            workflow_func (callable): The workflow function to test.
            test_case (TestCase): The test case containing input and expected output.
        """
        if isinstance(test_case, dict):
            test_case = TestCase.model_validate(test_case)
        result = None
        try:
            result = await workflow_func(*test_case.test_input)
        except Exception as e:
            if not test_case.expect_errors:
                raise
        self.validate_result(test_case, result)
        return result

    def test_workflow(self, workflow_func, test_case:Union[TestCase, dict]):
        """Run the workflow function with the test case input and validate the output.
        Args:
            workflow_func (callable): The workflow function to test.
            test_case (TestCase): The test case containing input and expected output.
        """
        if isinstance(test_case, dict):
            test_case = TestCase.model_validate(test_case)
        result = None
        try:
            result = workflow_func(*test_case.test_input)
        except Exception as e:
            if not test_case.expect_errors:
                raise
        self.validate_result(test_case, result)
        return result

    @staticmethod
    def run_agent(agent, agent_type:str, *args, **kwargs):
        agent_runner = get_agent_runner(agent_type)
        if agent_runner is None:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        result = agent_runner.run_agent(agent, *args, **kwargs)
        return result

    @staticmethod
    async def run_agent_async(agent, agent_type:str, *args, **kwargs):
        agent_runner = get_agent_runner(agent_type)
        if agent_runner is None:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        result = await agent_runner.run_agent_async(agent, *args, **kwargs)
        return result

    async def test_agent_async(self, agent, agent_type:str, test_case:Union[TestCase, dict]):
        if isinstance(test_case, dict):
            test_case = TestCase.model_validate(test_case)
        result = None
        try:
            result = await MonocleValidator.run_agent_async(agent, agent_type, *test_case.test_input)
        except Exception as e:
            if not test_case.expect_errors:
                raise
        self.validate_result(test_case, result)
        return result

    def test_agent(self, agent, agent_type:str, test_case:Union[TestCase, dict]):
        if isinstance(test_case, dict):
            test_case = TestCase.model_validate(test_case)
        result = None
        try:
            result = MonocleValidator.run_agent(agent, agent_type, *test_case.test_input)
        except Exception as e:
            if not test_case.expect_errors:
                raise
        self.validate_result(test_case, result)
        return result

    def _set_wrapper_methods(self, mock_tools: list[MockTool]) -> list[dict]:
        skip_exec: dict[str, dict] = {}
        context = None
        for mock_tool in mock_tools:
            skip_exec[mock_tool.name] = {
                "entity.type": mock_tool.type,
                "span.type": "agentic.tool.invocation",
                "response": mock_tool.response,
                "raise_error": mock_tool.raise_error,
                "error_message": mock_tool.error_message
            }
        if len(skip_exec) > 0:
            context = set_value(MONOCLE_SKIP_EXECUTIONS, skip_exec)
        return context

    def validate(self, test_case:TestCase) -> bool:
        """Validate the test case against the collected spans.
         Args:
            test_case (AgentTestCase): The test case to validate.
         """
        for test_span in test_case.test_spans:
            if test_span.span_type == SpanType.TOOL_INVOCATION:
                self.verify_tool_invoked(test_span)
            elif test_span.span_type == SpanType.AGENTIC_INVOCATION:
                self.verify_agent_invoked(test_span)
            elif test_span.span_type == SpanType.AGENTIC_REQUEST:
                self.verify_agentic_request(test_span)
            elif test_span.span_type == SpanType.AGENTIC_DELEGATION:
                from_agent = test_span.entities[0].name
                to_agent = test_span.entities[1].name
                self.verify_agent_delegated(from_agent, to_agent, test_span.positive_test, test_span.expect_errors, test_span.expect_warnings)
            elif test_span.span_type == SpanType.INFERENCE:
                self.verify_inference(test_span)

        if test_case.expect_errors:
            self._has_errors(test_case.expect_errors)
        if test_case.expect_warnings:
            self._has_warnings(test_case.expect_warnings)

        return True

    def validate_result(self, test_case:TestCase, result) -> bool:
        if test_case.test_output is not None:
            assert test_case.comparer.compare(test_case.test_output, result), "Result does not match expected output."
        return True

    def verify_agentic_request(self, test_span: TestSpan) -> bool:
        expected_request:str = test_span.input
        expected_response:str = test_span.output
        comparer:BaseComparer = test_span.comparer
        positive_test:bool = test_span.positive_test
        expect_errors:bool = test_span.expect_errors
        expect_warnings:bool = test_span.expect_warnings
        eval:Evaluation = test_span.eval

        agent_request_span = self._get_agent_request_span(expect_errors, expect_warnings)

        actual_output = self._agent_request_output(agent_request_span, expect_errors, expect_warnings)
        assert actual_output is not None, "No response found in agent request span."
        if expected_response is not None:
            valid_response:bool = self._valid_response(expected_response, actual_output.get("output"), comparer)
            if not valid_response and positive_test:
                assert False, f"Expected response doesn't match"
            elif valid_response and not positive_test:
                assert False, f"Response matched, but was not expected to be"
        if expected_request is not None:
            valid_input:bool = self._valid_response(expected_request, actual_output.get("input"), comparer)
            if not valid_input and positive_test:
                assert False, f"Expected request doesn't match"
            elif valid_input and not positive_test:
                assert False, f"Request matched, but was not expected to be"
        if eval is not None:
            self._evaluate_span(agent_request_span, eval, positive_test)

    def verify_inference(self, test_span: TestSpan) -> bool:
        """Verify that the inference response matches the expected response or schema.
         Args:
            inference_to_verify (InferenceToVerify): The inference to verify.
         """
        expected_output:Union[str, dict] = test_span.output
        expected_schema = None
        comparer = test_span.comparer
        positive_test = test_span.positive_test
        expect_errors = test_span.expect_errors
        expect_warnings = test_span.expect_warnings
        eval:Evaluation = test_span.eval
        max_output_tokens:Optional[int] = None

        inference_spans = self._get_inference_spans(expect_errors, expect_warnings)
        if len(inference_spans) == 0:
            assert False, f"No inferences found in spans."
        verified_schema:bool = False
        verified_response:bool = False
        verified_response_span:Span = None
        for inference_span in inference_spans:
            inference_response = self._get_inference_output(inference_span)
            # if expected schema is provided, validate against schema
            if expected_schema is not None and not verified_schema:
                if isinstance(expected_schema, dict):
                    try:
                        jsonschema.validate(instance=json.loads(inference_response), schema=expected_schema)
                    except jsonschema.ValidationError as e:
                        continue
                elif issubclass(expected_schema, BaseModel):
                    if isinstance(inference_response, str):
                        try:
                            expected_schema.model_validate_json(inference_response)
                        except ValidationError as e:
                            continue
                verified_schema = True
                if not expect_errors and self._span_has_error(inference_span):
                    assert False, f"Inference matched the expected schema but had errors."
                if not expect_warnings and self._span_has_warning(inference_span):
                    assert False, f"Inference matched the expected schema but had warnings."
            if expected_output is not None and not verified_response:
                if self._valid_response(expected_output, inference_response, comparer):
                    verified_response = True
                    if not expect_errors and self._span_has_error(inference_span):
                        assert False, f"Inference matched the expected schema but had errors."
                    if not expect_warnings and self._span_has_warning(inference_span):
                        assert False, f"Inference matched the expected schema but had warnings."
                    verified_response_span = inference_span

        if expected_schema is not None and not verified_schema and positive_test:
            assert False, f"No inference matched the expected schema."
        elif expected_schema is not None and verified_schema and not positive_test:
            assert False, f"An inference matched the expected schema, but was not expected to be."

        if expected_output is not None and not verified_response and positive_test:
            assert False, f"No inference matched the expected response."
        elif expected_output is not None and verified_response and not positive_test:
            assert False, f"An inference matched the expected response, but was not expected to be."

        if eval is not None and verified_response and verified_response_span is not None:
            self._evaluate_span(verified_response_span, eval, positive_test)

        if max_output_tokens is not None:
            self.check_completion_token_limits(max_output_tokens, positive_test)
        return True

    def _valid_response(self, response_to_evaluate:str, actual_response:str, comparer:BaseComparer) -> bool:
        """Verify that the agent response matches the expected response.
         Args:
            response_to_evaluate (str): The expected response from the agent.
            actual_response (str): The actual response from the agent. If none passed, then it checks the response of the root agent.
            agent_name (str, optional): The name of the agent to filter spans. If none passed, then checks response of the root agent. Defaults to None.
         """
        return comparer.compare(response_to_evaluate, actual_response)

    def verify_tool_invoked(self, test_span:TestSpan) -> bool:
        """Verify that a specific tool was invoked by the agent.
         Args:
            test_span (TestSpan): The tool invocation test span to verify.
         """
        tool_name:str = test_span.entities[0].name
        agent_name:str = test_span.entities[1].name if len(test_span.entities) > 1 else None
        tool_inputs:list[str] = [test_span.input] if test_span.input is not None else []
        tool_outputs:list[str] = [test_span.output] if test_span.output is not None else []
        positive_test:bool = test_span.positive_test
        expect_error:bool = test_span.expect_errors
        expect_warnings:bool = test_span.expect_warnings
        comparer:BaseComparer = test_span.comparer
        eval:Evaluation = test_span.eval

        tool_invocation_spans = self._get_tool_invocation_spans(tool_name, agent_name)
        if len(tool_invocation_spans) == 0:
            if positive_test:
                assert False, f"Tool '{tool_name}' was not invoked by agent '{agent_name}'."
            return True

        if not positive_test:
            assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}', but was not expected to be."

        self._check_error_warnings(tool_invocation_spans, tool_name, agent_name, expect_error, expect_warnings)
        self._check_input_output(tool_invocation_spans, tool_inputs, tool_outputs, comparer, eval, positive_test,
                                    tool_name=tool_name, agent_name=agent_name)

        return True

    def verify_agent_invoked(self, test_span: TestSpan) -> bool:
        """Verify that a specific agent was invoked.
         Args:
            agent_name (str): The name of the agent to verify invocation.
         """
        agent_name:str = test_span.entities[0].name
        positive_test:bool = test_span.positive_test
        expect_error:bool = test_span.expect_errors
        expect_warnings:bool = test_span.expect_warnings
        eval:Evaluation = test_span.eval
        comparer:BaseComparer = test_span.comparer
        agent_inputs:list[str] = [test_span.input] if test_span.input is not None else []
        agent_outputs:list[str] = [test_span.output] if test_span.output is not None else []

        agent_invocation_spans = self._get_agent_invocation_spans(agent_name)
        if len(agent_invocation_spans) == 0:
            if positive_test:
                assert False, f"Agent '{agent_name}' was not invoked."
            return True
        if not positive_test:
            assert False, f"Agent '{agent_name}' was invoked, but was not expected to be."
        self._check_error_warnings(agent_invocation_spans, None,agent_name, expect_error, expect_warnings)
        self._check_input_output(agent_invocation_spans, agent_inputs, agent_outputs, comparer, eval, positive_test, agent_name=agent_name)
        return True

    def verify_agent_delegated(self, from_agent:str, to_agent:str, positive_test:bool, expect_error:bool ,
                        expect_warnings:bool) -> bool:
        """Verify that a specific agent was delegated to.
         Args:
            from_agent (str): The name of the agent that delegated the task.
            to_agent (str): The name of the agent that was delegated the task.
         """
        found_delegation = False
        for span in self.spans:
            span_attributes = span.attributes
            if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.delegation"
                and span_attributes.get("entity.1.to_agent","") == to_agent
                and span_attributes.get("entity.1.from_agent","") == from_agent
            ):
                if not positive_test:
                    assert False, f"Agent '{to_agent}' was delegated by '{from_agent}', but was not expected to be."
                if not expect_error and self._span_has_error(span):
                    assert False, f"Agent '{to_agent}' was delegated by '{from_agent}' but error found."
                if expect_warnings and not self._span_has_warning(span):
                    assert False, f"Agent '{to_agent}' was delegated by '{from_agent}' warning was expected but no warning found."
                found_delegation = True

        if positive_test and not found_delegation:
            assert False, f"Agent '{to_agent}' was not delegated by '{from_agent}'."
        return True

    def check_completion_token_limits(self, max_output_tokens:int, positive_test:bool = True) -> bool:
        """Verify that the output token limits are respected in the spans.
         Args:
            max_tokens (int): The maximum number of tokens allowed in the output.
         """
        if max_output_tokens is None:
            return True
        return self._check_token_limits(max_output_tokens, "completion_tokens", positive_test)

    def check_total_token_limits(self, max_total_tokens:int, positive_test:bool = True) -> bool:
        """Verify that the output token limits are respected in the spans.
         Args:
            max_tokens (int): The maximum number of tokens allowed in the output.
         """
        if max_total_tokens is None:
            return True
        return self._check_token_limits(max_total_tokens, "total_tokens", positive_test)

    def _check_token_limits(self, max_tokens:int, token_name:str, positive_test:bool) -> bool:
        """Verify that the output token limits are respected in the spans.
         Args:
            max_tokens (int): The maximum number of tokens allowed in the output.
         """
        for span in self.spans:
            span_attributes = span.attributes
            if (
                "span.type" in span_attributes
                and (span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework")
            ):
                for event in span.events:
                    if event.name == "metadata":
                        tokens += event.attributes.get(token_name, 0)

        if positive_test:
            assert tokens <= max_tokens, f" {token_name} limit exceeded: {tokens} > {max_tokens}"
        else:
            assert tokens > max_tokens, f" {token_name} limit was not exceeded as expected: {tokens} <= {max_tokens}"
        return True

    def _verify_tool_errors(self, tool_name:str, agent_name:str, expect_error:bool, found_error: bool, expect_warnings:bool, found_warning: bool) -> None:
            if expect_error and not found_error:
                assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}' error was expected but no error found."
            elif not expect_error and found_error:
                assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}' but error found."
            if expect_warnings and not found_warning:
                assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}' warning was expected but no warning found."
            elif not expect_warnings and found_warning:
                assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}' but warning found."

    def _verify_tool_input_output(self, tool_name: str, agent_name: str, tool_inputs: Optional[list[str]], found_input: bool,
                    tool_outputs: Optional[list[str]], found_output: bool, positive_test: bool) -> None:
        if len(tool_inputs) > 0 and found_input is False and positive_test:
            assert False, f"Tool '{tool_name}' was never invoked by agent {agent_name} with input '{tool_inputs}'."
        elif len(tool_inputs) > 0 and found_input is True and not positive_test:
            assert False, f"Tool '{tool_name}' was invoked by agent {agent_name} with input '{tool_inputs}', but was not expected to be."

        if len(tool_outputs) > 0 and found_output is False and positive_test:
            assert False, f"Tool '{tool_name}' invoked by agent {agent_name} didn't return '{tool_outputs}'."
        elif len(tool_outputs) > 0 and found_output is True and not positive_test:
            assert False, f"Tool '{tool_name}' invoked by agent {agent_name} returned '{tool_outputs}', but was not expected to be."

    def _verify_agent_input_output(self, agent_name: str, agent_inputs: Optional[list[str]], found_input: bool,
                    agent_outputs: Optional[list[str]], found_output: bool, positive_test: bool) -> None:
        if len(agent_inputs) > 0 and found_input is False and positive_test:
            assert False, f"Agent '{agent_name}' was never invoked with input '{agent_inputs}'."
        elif len(agent_inputs) > 0 and found_input is True and not positive_test:
            assert False, f"Agent '{agent_name}' was invoked with input '{agent_inputs}', but was not expected to be."

        if len(agent_outputs) > 0 and found_output is False and positive_test:
            assert False, f"Agent '{agent_name}' didn't return '{agent_outputs}'."
        elif len(agent_outputs) > 0 and found_output is True and not positive_test:
            assert False, f"Agent '{agent_name}' returned '{agent_outputs}', but was not expected to be."

    def _check_error_warnings(self, spans:list[Span], tool_name:Optional[str], agent_name:str, expect_error:bool, expect_warnings:bool) -> None:
            found_error = False
            found_warning = False
            for span in spans:
                if self._span_has_error(span):
                    found_error = True
                if self._span_has_warning(span):
                    found_warning = True
            if tool_name is not None:
                self._verify_tool_errors(tool_name, agent_name, expect_error, found_error, expect_warnings, found_warning)
            else:
                self._verify_agent_errors(agent_name, expect_error, found_error, expect_warnings, found_warning)

    def _verify_agent_errors(self, agent_name:str, expect_error:bool, found_error: bool, expect_warnings:bool, found_warning: bool) -> None:
            if expect_error and not found_error:
                assert False, f"Agent '{agent_name}' was invoked error was expected but no error found."
            elif not expect_error and found_error:
                assert False, f"Agent '{agent_name}' was invoked but error found."
            if expect_warnings and not found_warning:
                assert False, f"Agent '{agent_name}' was invoked warning was expected but no warning found."
            elif not expect_warnings and found_warning:
                assert False, f"Agent '{agent_name}' was invoked but warning found."

    def _check_input_output(self, spans:list[Span], expected_inputs:Optional[list[str]], expected_outputs:Optional[list[str]],
                        comparer:BaseComparer, eval:Optional[Evaluation], positive_test:Optional[bool]=True,
                        tool_name:Optional[str]=None, agent_name:Optional[str]=None) -> None:
        candidate_spans = []
        found_input = found_output = False
        if len(expected_inputs) > 0 or len(expected_outputs) > 0:
            candidate_span = None
            for span in spans:
                found_input_in_span = False
                found_output_in_span = False
                for event in span.events:
                    if event.name == "data.input":
                        for expected_input in expected_inputs:
                            if comparer.compare(expected_input, event.attributes.get("input")):
                                found_input_in_span = True
                                break
                    elif event.name == "data.output":
                        for expected_output in expected_outputs:
                            if comparer.compare(expected_output, event.attributes.get("response")):
                                found_output_in_span = True
                                break
                if found_input_in_span and found_output_in_span:
                    found_input = found_output = True
                    candidate_span = span
                elif found_input_in_span and len(expected_outputs) == 0:
                    found_input = True
                    candidate_span = span
                elif found_output_in_span and len(expected_inputs) == 0:
                    found_output = True
                    candidate_span = span
                if candidate_span is not None:
                    if eval is not None:
                        self._evaluate_span(candidate_span, eval, positive_test)
                    candidate_spans.append(candidate_span)
            if tool_name is not None:
                self._verify_tool_input_output(tool_name, agent_name, expected_inputs, found_input, expected_outputs, found_output, positive_test)
            elif agent_name is not None:
                self._verify_agent_input_output(agent_name, expected_inputs, found_input, expected_outputs, found_output, positive_test)
        return candidate_spans

    def _get_inference_spans(self) -> list[Span]:
        inferences: list[Span] = []
        for span in self.spans:
            span_attributes = span.attributes
            if (
                "span.type" in span_attributes
                and (span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework")
            ):
                inferences.append(span)
        return inferences

    def _get_inference_output(self, span) -> Union[str, dict, list]:
        for event in span.events:
            if event.name == "data.output":
                return event.attributes.get("response", "")
        return ""

    def _get_output_response(self, span) -> Union[str, dict, list]:
        for event in span.events:
            if event.name == "data.output":
                return event.attributes.get("response", "")
        return ""

    def _get_agent_request_span(self, expect_error:bool = False, expect_warnings: bool = False) -> Optional[Span]:
        for span in self.spans:
            span_attributes = span.attributes
            if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.turn"
            ):
                found_error = self._span_has_error(span)
                if expect_error and not found_error:
                    assert False, f"Agent request span error was expected but no error found."
                elif not expect_error and found_error:
                    assert False, f"Agent request span error found."
                
                found_warning = self._span_has_warning(span)
                if expect_warnings and not found_warning:
                    assert False, f"Agent request span warning was expected but no warning found."
                elif not expect_warnings and found_warning:
                    assert False, f"Agent request span warning found."
                return span
        return None

    def _get_tool_invocation_spans(self, tool_name:str, agent_name:str = None) -> list:
        tool_invocation_spans = []
        for span in self.spans:
            span_attributes = span.attributes
            if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.tool.invocation"
            ):
                if span_attributes.get("entity.1.name","") == tool_name \
                    and (agent_name is None or (agent_name is not None and span_attributes.get("entity.2.name","") == agent_name)):
                    tool_invocation_spans.append(span)
        return tool_invocation_spans
    
    def _get_agent_invocation_spans(self, agent_name:str) -> list:
        agent_invocation_spans = []
        for span in self.spans:
            span_attributes = span.attributes
            if (
                "span.type" in span_attributes
                and span_attributes["span.type"] == "agentic.invocation"
            ):
                if span_attributes.get("entity.1.name","") == agent_name:
                    agent_invocation_spans.append(span)
        return agent_invocation_spans

    def _agent_request_output(self, agent_request_span:Span,expect_error:bool = False, expect_warnings: bool = False) -> dict[str, str]:
        if agent_request_span is None:
            return None
        input = output = None
        for event in agent_request_span.events:
            if event.name == "data.input":
                input = event.attributes.get("input", "")
            elif event.name == "data.output":
                output = event.attributes.get("response", "")
        return {"input": input, "output": output}

    def _has_errors(self, expect_errors: bool) -> bool:
        found_error = False
        for span in self.spans:
            if self._span_has_error(span):
                if not expect_errors:
                    assert False, f"Span {span.name} have error status."
                found_error = True
        if expect_errors and not found_error:
            assert False, f"Span {span.name} does not have error status."
        return True

    def _span_has_error(self, span:Span) -> bool:
        if span.status.status_code == StatusCode.ERROR:
            return True
        return False
    
    def _has_warnings(self, expect_warnings: bool) -> bool:
        found_warning = False
        for span in self.spans:
            if self._span_has_error(span):
                continue
            if self._span_has_warning(span):
                if not expect_warnings:
                    assert False, f"Span {span.name} has warning event."
                found_warning = True
        if expect_warnings and not found_warning:
            assert False, f"Span {span.name} does not have warning event."
        return True

    def _span_has_warning(self, span:Span) -> bool:
        for event in span.events:
            if event.name == "metadata":
                finish_type = event.attributes.get("finish_type", "success")
                if finish_type != "success":
                    return True
        return False

    def _evaluate_span(self, span:Span, evaluation:Evaluation,  positive_test:bool) -> None:
        eval_args = {}
        for arg in evaluation.args:
            if arg == EvalInputs.INPUT:
                eval_args['input'] = trace_utils.get_input_from_span(span)
            elif arg == EvalInputs.OUTPUT:
                eval_args['output'] = trace_utils.get_output_from_span(span)
            elif arg == EvalInputs.AGENT_DESCRIPTION:
                eval_args['agent_description'] = trace_utils.get_agent_description_from_span(span)
        actual_eval: dict = evaluation.eval.evaluate(eval_args)
        if positive_test and not evaluation.comparer.compare(evaluation.expected_result, actual_eval):
            assert False, f"Span {span.name} evaluation failed. Expected: {evaluation.expected_result}, Actual: {actual_eval}"
        elif not positive_test and evaluation.comparer.compare(evaluation.expected_result, actual_eval):
            assert False, f"Span {span.name} evaluation passed, but was not expected to. Actual: {actual_eval}"
