"""The scenario spec: the JSON a caller writes to describe one test scenario."""
import re
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ScenarioParam(BaseModel):
    """One detail of the request. Becomes a zero-arg tool on the test agent, so a
    withheld value can only reach the conversation when the target asks for it."""

    name: str = Field(..., description="Param name; becomes the tool name.")
    value: Any = Field(..., description="The value the tool returns.")
    description: str = Field(..., description="Becomes the tool description.")
    is_initial: bool = Field(False, description="Revealed in the opening message.")
    required: bool = Field(
        True,
        description="The target agent must ask for this detail. Auto-satisfied when "
                    "is_initial, since the value goes out in the opening message.")

    @field_validator("name")
    @classmethod
    def _identifier_safe(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError(
                f"param name '{value}' must be a valid Python identifier "
                "because it becomes a tool name"
            )
        return value


class ScenarioTestCase(BaseModel):
    """A single test scenario: what to test, as whom, with what details."""

    test_name: str = Field("monocle_scenario_test", description="Name of the scenario.")
    scenario: str = Field(..., description="What is being tested.")
    persona: str = Field(..., description="User persona the test agent emulates.")
    params: list[ScenarioParam] = Field(default_factory=list,
                                        description="Details of the request.")
    success_criteria: str = Field(..., description="Judged by the response evaluator.")
    max_turns: int = Field(10, description="Maximum target-agent invocations.")
    min_turns: int = Field(1, description="The criteria do not count as met before this turn, so a "
                                          "pressure scenario plays out; a violation still ends it at once.")
    session_id: Optional[str] = Field(None, description="Auto-generated when omitted.")

    @field_validator("scenario", "success_criteria")
    @classmethod
    def _not_blank(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_case(self) -> "ScenarioTestCase":
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if not 1 <= self.min_turns <= self.max_turns:
            raise ValueError("min_turns must be between 1 and max_turns")
        names = [param.name for param in self.params]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate param names: {duplicates}")
        if self.params and not any(param.is_initial for param in self.params):
            raise ValueError(
                "at least one param must have is_initial=True, otherwise the test "
                "agent has nothing to open the conversation with"
            )
        if self.session_id is None:
            self.session_id = f"monocle_scenario_session_{uuid.uuid4().hex}"
        return self

    def missing_required_params(self, requested: set[str]) -> list[str]:
        """Required withheld params whose tool was never called: the target never asked."""
        return [param.name for param in self.params
                if param.required and not param.is_initial
                and param.name not in requested]

    @property
    def initial_params(self) -> list[ScenarioParam]:
        """Params stated up front in the opening message."""
        return [param for param in self.params if param.is_initial]

    @property
    def on_request_params(self) -> list[ScenarioParam]:
        """Params revealed only when the target agent asks for them."""
        return [param for param in self.params if not param.is_initial]
