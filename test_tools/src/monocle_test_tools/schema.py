from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field
from monocle_test_tools.evals.base_eval import BaseEval
from monocle_test_tools.evals.bert_eval import BertScorerEval
from monocle_test_tools.comparer.base_comparer import BaseComparer
from monocle_test_tools.comparer.bert_score_comparer import BertScoreComparer
from monocle_test_tools.comparer.default_comparer import DefaultComparer

class SpanType(str, Enum):
    TOOL_INVOCATION = "agentic.tool.invocation"
    AGENTIC_INVOCATION = "agentic.invocation"
    AGENTIC_REQUEST = "agentic.request"
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

class TestSpan(BaseModel):
    span_type: SpanType = Field(..., description="Type of the span.")
    entities: Optional[list[Entity]] = Field(None, description="List of entities involved in the span.")
    input: Optional[Union[str, dict]] = Field(None, description="Input for the span.")
    output: Optional[Union[str, dict]] = Field(None, description="Output for the span.")
    test_type: Optional[str] = Field("positive", description="Whether the test is positive or negative, default is positive.")
    eval: Optional[Union[str, BaseEval]] = Field(None, description="Evaluation method for the span.")
    eval_options: Optional[dict] = Field({}, description="Options for the evaluation method.")
    expect_errors: Optional[bool] = Field(False, description="Whether to expect errors during the test.")
    expect_warnings: Optional[bool] = Field(False, description="Whether to expect warnings during the test.")
    comparer: Optional[Union[str, BaseComparer]] = Field(DefaultComparer(), description="Comparison method for the span.")
    positive_test: Optional[bool] = True


    def __init__(self, **data):
        super().__init__(**data)
        if self.span_type in {SpanType.TOOL_INVOCATION, SpanType.AGENTIC_INVOCATION, SpanType.AGENTIC_REQUEST, SpanType.AGENTIC_DELEGATION} and (self.entities is None or len(self.entities) == 0):
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
        if isinstance(self.eval, str):
            try:
                # instantiate eval class from string
                eval_class = globals()[self.eval]
                if issubclass(eval_class, BaseEval):
                    self.eval = eval_class(eval_options=self.eval_options)
            except Exception as e:
                raise ValueError(f"Invalid eval class name: {self.eval}. Error: {e}")
        if isinstance(self.comparer, str):
            if self.comparer == "default":
                self.comparer = DefaultComparer()
            elif self.comparer == "bert_score":
                self.comparer = BertScoreComparer()
            else:
                try:
                    # instantiate comparer class from string
                    comparer_class = globals()[self.comparer]
                    if issubclass(comparer_class, BaseComparer):
                        self.comparer = comparer_class(eval_options=self.eval_options)
                except Exception as e:
                    raise ValueError(f"Invalid comparer class name: {self.comparer}. Error: {e}")
        if self.test_type == "positive":
            self.positive_test = True
        elif self.test_type == "negative":
            self.positive_test = False
        else:
            raise ValueError("test_type must be either 'positive' or 'negative'.")


class TestCase(BaseModel):
    test_input: Optional[str] = Field(None, description="Input prompt or data for the test case.")
    test_description: Optional[str] = Field(None, description="Description of the test case.")
    test_spans: list[TestSpan] = Field(default_factory=list, description="List of spans to include in the test case.")
    expect_errors: Optional[bool] = Field(False, description="Whether to expect errors during the test.")
    expect_warnings: Optional[bool] = Field(False, description="Whether to expect warnings during the test.")
