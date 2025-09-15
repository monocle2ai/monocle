from functools import wraps
import inspect
import jsonschema, json
from typing import Optional, Union
from opentelemetry.sdk.trace import Span, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import pytest
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from contextlib import contextmanager, asynccontextmanager
import logging
from monocle_apptrace.instrumentation.common.instrumentor import MonocleInstrumentor, setup_monocle_telemetry
from pydantic import BaseModel, ValidationError
from monocle_test_tools.schema import SpanType, TestSpan, TestCase
from monocle_test_tools.comparer.base_comparer import BaseComparer

logger = logging.getLogger(__name__)

class MonocleValidator:
    _spans:Span = []
    memory_exporter:InMemorySpanExporter = None
    file_exporter:FileSpanExporter = None
    trace_id = None
    instrumentor: MonocleInstrumentor = None
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if MonocleValidator._initialized:
            return
        self.memory_exporter = InMemorySpanExporter()
        self.file_exporter = FileSpanExporter()
        span_processors = [SimpleSpanProcessor(self.file_exporter), SimpleSpanProcessor(self.memory_exporter)]
        self.instrumentor = setup_monocle_telemetry(workflow_name="monocle_validator", span_processors=span_processors)
        MonocleValidator._initialized = True

    @property
    def spans(self):
        if len(self._spans) == 0 and self.memory_exporter is not None:
            self._spans = self.memory_exporter.get_finished_spans()
        return self._spans

    @contextmanager
    def monocle_exporter_wrapper(self, test_case: TestCase, request):
        test_case_name = request.node.name if request is not None else (test_case.test_case_name if test_case is not None else "monocle_test")
        self.file_exporter.set_service_name(test_case_name)
        try:
            yield
        finally:
            try:
                self.validate(test_case)
            finally:
                self.memory_exporter.clear()
                self.file_exporter.force_flush()
                self.file_exporter.shutdown()
                self._spans = []

    def monocle_testcase(self, test_cases: list[TestCase]):
        """Decorator to mark a test function as a monocle test case."""
        def decorator(func):
            if inspect.iscoroutinefunction(func):
#                @wraps(func)
                @pytest.mark.asyncio
                @pytest.mark.parametrize("test_case", test_cases)
                async def wrapper(test_case, request, *args, **kwargs):
                    with self.monocle_exporter_wrapper(test_case, request):
                        return await func(test_case, *args, **kwargs)
            else:
#                @wraps(func)
                @pytest.mark.parametrize("test_case", test_cases)
                def wrapper(test_case, request, *args, **kwargs):
                    with self.monocle_exporter_wrapper(test_case, request):
                        return func(test_case, *args, **kwargs)
            return wrapper
        return decorator

    def validate(self, test_case:TestCase) -> bool:
        """Validate the test case against the collected spans.
         Args:
            test_case (AgentTestCase): The test case to validate.
         """
        for test_span in test_case.test_spans:
            if test_span.span_type == SpanType.TOOL_INVOCATION:
                self.tool_invoked(test_span.entities[0].name, test_span.entities[1].name if len(test_span.entities) > 1 else None,
                                  test_span.input, test_span.output, test_span.positive_test, test_span.expect_errors, test_span.expect_warnings)
            elif test_span.span_type == SpanType.AGENTIC_INVOCATION:
                self.agent_invoked(test_span.entities[0].name, test_span.positive_test, test_span.expect_errors, test_span.expect_warnings)
            elif test_span.span_type == SpanType.AGENTIC_REQUEST:
                self.verify_agentic_request(test_span.input, test_span.output, test_span.positive_test, test_span.comparer,
                                    test_span.expect_errors, test_span.expect_warnings)
            elif test_span.span_type == SpanType.AGENTIC_DELEGATION:
                from_agent = test_span.entities[0].name
                to_agent = test_span.entities[1].name
                self.agent_delegated(from_agent, to_agent, test_span.positive_test, test_span.expect_errors, test_span.expect_warnings)
            elif test_span.span_type == SpanType.INFERENCE:
                self.verify_inference(test_span.output, None, test_span.comparer, test_span.positive_test, test_span.comparer, test_span.expect_errors, test_span.expect_warnings)

        if test_case.expect_errors:
            self._has_errors(test_case.expect_errors)
        if test_case.expect_warnings:
            self._has_warnings(test_case.expect_warnings)

        return True

    def verify_agentic_request(self, expected_request:str, expected_response:str, comparer:BaseComparer,
                            positive_test:bool, expect_errors: bool, expect_warnings: bool) -> bool:
        actual_output = self._agent_request_output(expect_errors, expect_warnings)
        assert actual_output is not None, "No response found in agent request span."
        if expected_response is not None:
            valid_response:bool = self.valid_response(expected_response, actual_output, comparer)
            if not valid_response and positive_test:
                assert False, f"Expected response doesn't match"
            elif valid_response and not positive_test:
                assert False, f"Response matched, but was not expected to be"
        if expected_request is not None:
            valid_input:bool = self.valid_response(expected_request, actual_output, comparer)
            if not valid_input and positive_test:
                assert False, f"Expected request doesn't match"
            elif valid_input and not positive_test:
                assert False, f"Request matched, but was not expected to be"

    def verify_inference(self, expected_output:Union[str, dict], expected_schema:Union[dict, BaseModel],
                    comparer:BaseComparer, positive_test:bool, expect_errors: bool,
                    expect_warnings:bool , max_output_tokens:Optional[int] = None) -> bool:
        """Verify that the inference response matches the expected response or schema.
         Args:
            inference_to_verify (InferenceToVerify): The inference to verify.
         """
        inference_spans = self._get_inference_spans(expect_errors, expect_warnings)
        if len(inference_spans) == 0:
            assert False, f"No inferences found in spans."
        verified_schema:bool = False
        verified_response:bool = False
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
                if self.valid_response(expected_output, inference_response, comparer):
                    verified_response = True
                    if not expect_errors and self._span_has_error(inference_span):
                        assert False, f"Inference matched the expected schema but had errors."
                    if not expect_warnings and self._span_has_warning(inference_span):
                        assert False, f"Inference matched the expected schema but had warnings."

        if expected_schema is not None and not verified_schema and positive_test:
            assert False, f"No inference matched the expected schema."
        elif expected_schema is not None and verified_schema and not positive_test:
            assert False, f"An inference matched the expected schema, but was not expected to be."

        if expected_output is not None and not verified_response and positive_test:
            assert False, f"No inference matched the expected response."
        elif expected_output is not None and verified_response and not positive_test:
            assert False, f"An inference matched the expected response, but was not expected to be."

        if max_output_tokens is not None:
            self.check_token_limits(max_output_tokens, positive_test)
        return True

    def valid_response(self, response_to_evaluate:str, actual_response:str, comparer:BaseComparer) -> bool:
        """Verify that the agent response matches the expected response.
         Args:
            response_to_evaluate (str): The expected response from the agent.
            actual_response (str): The actual response from the agent. If none passed, then it checks the response of the root agent.
            agent_name (str, optional): The name of the agent to filter spans. If none passed, then checks response of the root agent. Defaults to None.
         """
        return comparer.compare(response_to_evaluate, actual_response)

    def tool_invoked(self, tool_name:str, agent_name:str = None, tool_input:str = None, tool_output:str = None,
                    positive_test:bool = True, expect_error:bool = False, expect_warnings:bool = False) -> bool:
        """Verify that a specific tool was invoked by the agent.
         Args:
            tool_name (str): The name of the tool to verify invocation.
            agent_name (str, optional): The name of the agent that invoked the tool. If none passed, then it matches any agent. Defaults to None.
         """
        tool_invocation_spans = self._get_tool_invocation_spans(tool_name, agent_name)
        if len(tool_invocation_spans) == 0:
            if positive_test:
                assert False, f"Tool '{tool_name}' was not invoked by agent '{agent_name}'."
        else:
            if not positive_test:
                assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}', but was not expected to be."

            found_error = False
            found_warning = False
            for span in tool_invocation_spans:
                if self._span_has_error(span):
                    found_error = True
                if self._span_has_warning(span):
                    found_warning = True
            if expect_error and not found_error:
                assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}' error was expected but no error found."
            elif not expect_error and found_error:
                assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}' but error found."
            if expect_warnings and not found_warning:
                assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}' warning was expected but no warning found."
            elif not expect_warnings and found_warning:
                assert False, f"Tool '{tool_name}' was invoked by agent '{agent_name}' but warning found."

        found_input = found_output = False
        if tool_input is not None or tool_output is not None:
            for span in tool_invocation_spans:
                for event in span.events:
                    if event.name == "data.input":
                        if tool_input is not None:
                            if event.attributes.get("input") == tool_input:
                                found_input = True
                    elif event.name == "data.output":
                        if tool_output is not None:
                            if event.attributes.get("response") == tool_output:
                                found_output = True
        if tool_input is not None and found_input is False and positive_test:
            assert False, f"Tool '{tool_name}' was never invoked by agent {agent_name} with input '{tool_input}'."
        elif tool_input is not None and found_input is True and not positive_test:
            assert False, f"Tool '{tool_name}' was invoked by agent {agent_name} with input '{tool_input}', but was not expected to be."

        if tool_output is not None and found_output is False and positive_test:
            assert False, f"Tool '{tool_name}' invoked by agent {agent_name} didn't return '{tool_output}'."
        elif tool_output is not None and found_output is True and not positive_test:
            assert False, f"Tool '{tool_name}' invoked by agent {agent_name} returned '{tool_output}', but was not expected to be."
        return True

    def agent_invoked(self, agent_name:str, positive_test:bool = True, expect_error:bool = False, expect_warnings:bool = False) -> bool:
        """Verify that a specific agent was invoked.
         Args:
            agent_name (str): The name of the agent to verify invocation.
         """
        agent_invocation_spans = self._get_agent_invocation_spans(agent_name)
        if len(agent_invocation_spans) == 0:
            if positive_test:
                assert False, f"Agent '{agent_name}' was not invoked."
        else:
            if not positive_test:
                assert False, f"Agent '{agent_name}' was invoked, but was not expected to be."
            found_error = False
            found_warning = False
            for span in agent_invocation_spans:
                if self._span_has_error(span):
                    found_error = True
                if self._span_has_warning(span):
                    found_warning = True
            if expect_error and not found_error:
                assert False, f"Agent '{agent_name}' was invoked error was expected but no error found."
            elif not expect_error and found_error:
                assert False, f"Agent '{agent_name}' was invoked but error found."
            if expect_warnings and not found_warning:
                assert False, f"Agent '{agent_name}' was invoked warning was expected but no warning found."
            elif not expect_warnings and found_warning:
                assert False, f"Agent '{agent_name}' was invoked but warning found."
        return True

    def agent_delegated(self, from_agent:str, to_agent:str, positive_test:bool = True, expect_error:bool = False,
                        expect_warnings:bool = False) -> bool:
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

    def check_token_limits(self, max_output_tokens:int, positive_test:bool = True) -> bool:
        """Verify that the output token limits are respected in the spans.
         Args:
            max_tokens (int): The maximum number of tokens allowed in the output.
         """
        if max_output_tokens is None:
            return True
        tokens = 0
        for span in self.spans:
            span_attributes = span.attributes
            if (
                "span.type" in span_attributes
                and (span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework")
            ):
                for event in span.events:
                    if event.name == "metadata":
                        tokens += event.attributes.get("completion_tokens", 0)

        if positive_test:
            assert tokens <= max_output_tokens, f" Output token limit exceeded: {tokens} > {max_output_tokens}"
        else:
            assert tokens > max_output_tokens, f" Output token limit was not exceeded as expected: {tokens} <= {max_output_tokens}"
        return True

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
                and span_attributes["span.type"] == "agentic.request"
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

    def _agent_request_output(self, expect_error:bool = False, expect_warnings: bool = False) -> str:
        agent_request_span = self._get_agent_request_span(expect_error, expect_warnings)
        if agent_request_span is None:
            return None
        output_event = agent_request_span.events[1]
        return output_event.attributes.get("response", "")

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
        if span.status.status_code == 'ERROR':
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