"""One scenario: what the user wants, who they are, the details they bring, and what success is."""
import re
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ScenarioParam(BaseModel):
    """One detail of the request. It becomes a tool on the simulated user, named ``name``."""

    name: str
    value: Any
    description: str                 # worded the way the agent would ask for it
    is_initial: bool = False         # stated up front; otherwise held back until asked
    required: bool = True            # a held-back detail the agent must ask for

    @field_validator("name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"param name '{value}' must be a valid Python identifier, because it becomes a tool name")
        return value


class ScenarioTestCase(BaseModel):
    test_name: str = "monocle_scenario_test"
    scenario: str = Field(..., min_length=1)
    persona: str
    params: list[ScenarioParam] = Field(default_factory=list)
    success_criteria: Optional[str] = Field(None, min_length=1)   # None: nothing is judged
    max_turns: int = Field(10, ge=1)
    min_turns: int = Field(1, ge=1)  # the criteria don't count as met before this turn
    session_id: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "ScenarioTestCase":
        if self.min_turns > self.max_turns:
            raise ValueError("min_turns must be between 1 and max_turns")
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate param names: {sorted({n for n in names if names.count(n) > 1})}")
        if self.params and not self.initial_params:
            raise ValueError("at least one param must have is_initial=True, so the user has something to open with")
        self.session_id = self.session_id or f"monocle_scenario_session_{uuid.uuid4().hex}"
        return self

    @property
    def initial_params(self) -> list[ScenarioParam]:
        return [p for p in self.params if p.is_initial]

    @property
    def on_request_params(self) -> list[ScenarioParam]:
        return [p for p in self.params if not p.is_initial]

    def missing_required_params(self, requested: set[str]) -> list[str]:
        """Held-back, required details the agent never asked for."""
        return [p.name for p in self.on_request_params if p.required and p.name not in requested]
