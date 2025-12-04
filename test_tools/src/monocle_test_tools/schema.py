from enum import Enum
from typing import Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
from monocle_test_tools.evals.base_eval import BaseEval
from monocle_test_tools.evals.bert_eval import BertScorerEval
from monocle_test_tools.comparer.base_comparer import BaseComparer
from monocle_test_tools.comparer.bert_score_comparer import BertScoreComparer
from monocle_test_tools.comparer.default_comparer import DefaultComparer
from monocle_test_tools.comparer.metric_comparer import MetricComparer
from monocle_test_tools.evals.eval_manager import get_evaluator
from monocle_test_tools.comparer.comparer_manager import get_comparer

class SpanType(str, Enum):
    TOOL_INVOCATION = "agentic.tool.invocation"
    AGENTIC_INVOCATION = "agentic.invocation"
    AGENTIC_REQUEST = "agentic.turn"
    AGENTIC_DELEGATION = "agentic.delegation"
    INFERENCE = "inference"

class EntityType(str, Enum):
    TOOL = "tool"
    AGENT = "agent"
    INFERENCE = "inference"

class Entity(BaseModel):
    type: EntityType = Field(..., description="Type of the entity, e.g., tool, agent, inference.")
    name: str = Field(..., description="Name of the entity.")
    options: Optional[dict] = Field(None, description="Additional options for the entity.")

class EvalInputs(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    AGENT_DESCRIPTION = "agent_description"

class Evaluation(BaseModel):
    eval: Union[str, BaseEval] = Field(..., description="Evaluation method for the span.")
    eval_options: Optional[dict] = Field({}, description="Options for the evaluation method.")
    args: list[EvalInputs] = Field([EvalInputs.OUTPUT], description="Arguments for the evaluation method.")
    expected_result: dict = Field({}, description="Expected result from the evaluation.")
    comparer: Optional[Union[str, BaseComparer]] = Field(DefaultComparer(), description="Comparison method for the evaluation.")
    def __init__(self, **data):
        super().__init__(**data)
        if self.expected_result == {}:
            raise ValueError("expected_result must be provided and cannot be empty.")
        self.comparer = get_comparer(self.comparer)
        self.eval = get_evaluator(self.eval, self.eval_options or {})

class TestSpan(BaseModel):
    """
    Represents a specific interaction or event within a test case in the Monocle testing framework.
    
    A TestSpan defines a single testable unit of interaction such as a tool invocation,
    agent delegation, or inference process. Each span captures the entities involved,
    inputs and outputs, and validation criteria for that specific interaction.
    
    Test spans enforce specific validation rules based on their type. For example:
    - Tool invocation spans must include at least one tool entity
    - Agentic delegation spans must include at least two agent entities (delegator and delegatee)
    - Agentic invocation spans must include at least one agent entity
    
    Attributes:
        span_type: Type of interaction this span represents (e.g., tool invocation, agent delegation)
        entities: List of entities (tools, agents) involved in this interaction
        input: Input provided to this interaction
        output: Expected output from this interaction
        test_type: Whether this is a "positive" (expected to succeed) or "negative" (expected to fail) test
        eval: Evaluation method to apply to the span's results
        expect_errors: Flag indicating whether errors are expected during this span
        expect_warnings: Flag indicating whether warnings are expected during this span
        comparer: Method used to compare actual results with expected results
    """
    span_type: SpanType = Field(..., description="Type of the span.")
    entities: Optional[list[Entity]] = Field(None, description="List of entities involved in the span.")
    input: Optional[Union[str, dict]] = Field(None, description="Input for the span.")
    output: Optional[Union[str, dict]] = Field(None, description="Output for the span.")
    test_type: Optional[str] = Field("positive", description="Whether the test is positive or negative, default is positive.")
    eval: Optional[Evaluation] = Field(None, description="Evaluation method for the span.")
    expect_errors: Optional[bool] = Field(False, description="Whether to expect errors during the test.")
    expect_warnings: Optional[bool] = Field(False, description="Whether to expect warnings during the test.")
    comparer: Optional[Union[str, BaseComparer]] = Field(DefaultComparer(), description="Comparison method for the span.")
    positive_test: Optional[bool] = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.span_type in {SpanType.TOOL_INVOCATION, SpanType.AGENTIC_INVOCATION, SpanType.AGENTIC_DELEGATION} and (self.entities is None or len(self.entities) == 0):
            raise ValueError(f"{self.span_type} span must have at least one entity.")
        if self.span_type == SpanType.AGENTIC_DELEGATION:
            if (self.entities is None or len(self.entities) < 2):
                raise ValueError("Agentic delegation span must have at least two entities: the delegator and the delegatee.")
            if self.entities[0].type != EntityType.AGENT:
                raise ValueError("The first entity in an agentic delegation span must be of type 'agent' (the delegator).")
            if self.entities[1].type != EntityType.AGENT:
                raise ValueError("The second entity in an agentic delegation span must be of type 'agent' (the delegatee).")
        if self.span_type == SpanType.AGENTIC_INVOCATION:
            if (self.entities is None or len(self.entities) < 1):
                raise ValueError("Agentic invocation span must have at least one entity: the invoking agent.")
            if self.entities[0].type != EntityType.AGENT:
                raise ValueError("The first entity in an agentic invocation span must be of type 'agent'.")
        if self.span_type == SpanType.TOOL_INVOCATION:
            if (self.entities is None or len(self.entities) < 1):
                raise ValueError("Tool invocation span must have at least one entity: the invoked tool.")
            if self.entities[0].type != EntityType.TOOL:
                raise ValueError("The first entity in a tool invocation span must be of type 'tool'.")
            if len(self.entities) > 1 and self.entities[1].type != EntityType.AGENT:
                raise ValueError("If present, the second entity in a tool invocation span must be of type 'agent' (the invoking agent).")
        self.comparer = get_comparer(self.comparer)
        if self.test_type == "positive":
            self.positive_test = True
        elif self.test_type == "negative":
            self.positive_test = False
        else:
            raise ValueError("test_type must be either 'positive' or 'negative'.")

class ToolType(str, Enum):
    OPENAI = "tool.openai"
    ADK = "tool.adk"
    LLAMA_INDEX = "tool.llama_index"
    LANGGRAPH = "tool.langgraph"
    STRANDS = "tool.strands"

class MockTool(BaseModel):
    """
    Represents a mock tool implementation for testing purposes in the Monocle testing framework.
    
    A MockTool defines a simulated version of a tool that can be used in test cases to
    mimic the behavior of real tools without invoking external dependencies. This allows
    for controlled testing of agent interactions and logic.
    
    Attributes:
        name: Name of the mock tool.
        description: Description of what the mock tool does.
        response: Predefined response that the mock tool will return when invoked.
    """
    name: str = Field(..., description="Name of the tool.")
    type: ToolType = Field(..., description="Type of the tool.")
    raise_error: Optional[bool] = Field(False, description="Whether the mock tool should simulate an error when invoked.")
    error_message: Optional[str] = Field("Simulated tool error", description="Error message to raise if raise_error is True.")
    response: Optional[Any] = Field(..., description="Predefined response from the mock tool. Placeholder for tool input are supported.")

class TestCase(BaseModel):
    """
    Represents a test case in the Monocle testing framework.
    
    A TestCase defines the input, expected output, and evaluation criteria for testing
    AI agent behaviors. It can contain multiple test spans representing different 
    interaction points (tool invocations, agent delegations, etc.) within the test.
    
    Each test case can specify comparison methods for evaluating test results against 
    expected outcomes and can be configured to expect certain errors or warnings.
    
    Attributes:
        test_input: Input data provided to the test case, can be a prompt or structured data.
        test_output: Expected output that the test should produce.
        comparer: Method used to compare actual results with expected results.
        test_description: Human-readable description of what the test case verifies.
        test_spans: Collection of TestSpan objects defining specific interactions to test.
        expect_errors: Flag indicating whether the test should expect errors to occur.
        expect_warnings: Flag indicating whether the test should expect warnings to occur.
    """
    test_name: Optional[str] = Field("monocle_test", description="Name of the test case.")
    test_input: Optional[Tuple[Any, ...]] = Field(None, description="Input prompt or data for the test case.")
    test_output: Optional[Any] = Field(None, description="Expected output for the test case.")
    comparer: Optional[Union[str, BaseComparer]] = Field(DefaultComparer(), description="Comparison method for the test case.")
    test_description: Optional[str] = Field(None, description="Description of the test case.")
    test_spans: list[TestSpan] = Field(default_factory=list, description="List of spans to include in the test case.")
    expect_errors: Optional[bool] = Field(False, description="Whether to expect errors during the test.")
    expect_warnings: Optional[bool] = Field(False, description="Whether to expect warnings during the test.")
    mock_tools: Optional[list[MockTool]] = Field([], description="List of mock tool implementations.")

    def __init__(self, **data):
        super().__init__(**data)
        self.comparer = get_comparer(self.comparer)