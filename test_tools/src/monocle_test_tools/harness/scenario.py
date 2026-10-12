"""The scenario spec: the JSON a caller writes to describe one test scenario."""
import re
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ScenarioParam(BaseModel):
    """One detail of the request the test agent is making of the target agent.

    Each param becomes a zero-arg tool on the test agent, which is why ``name`` has
    to be identifier-safe. ``description`` becomes the tool description -- word it the
    way the target agent would refer to the detail, so the test agent can match the
    target's question to the right tool.

    ``is_initial`` params are stated in the opening message. The values of the rest are
    withheld from the prompt entirely and reach the test agent only through a tool call,
    which is what makes progressive disclosure structural rather than a matter of the
    model following instructions.
    """

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
        """Required on-request params the target agent never asked for.

        ``requested`` is the set of param names whose tools the test agent called --
        it calls one only when the target agent asks for that detail, so a name absent
        from the set means the target never asked. Initial params are exempt: their
        values are stated in the opening message, so nothing is left for the target to
        ask for. Names come back in declaration order.

        A non-empty result fails the scenario even when the judge is satisfied, which
        is what catches an agent that meets the success criteria by inventing a detail
        instead of asking for it.
        """
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
