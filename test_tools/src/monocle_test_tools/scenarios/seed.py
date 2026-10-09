"""The seed file: what the agent is, what people come to it to do, what it refuses.

Two forms, same file. The user writes the minimal one:

    agent: A travel booking agent that books flights and hotels.
    goals:
      - Book a flight from Bombay to Hyderabad on 22 Oct 2026
      - Book a hotel in Mumbai for 27 Nov 2026
    out_of_scope: [refunds, visa advice]

``expand`` turns each goal sentence into a goal template and its params with one
model call and writes the expanded form, which is what the generator reads and what
the user edits if the split came out wrong.
"""
import json
import os
import re
from typing import Any, Union

from pydantic import BaseModel, Field, field_validator, model_validator

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SeedError(ValueError):
    """Raised for an invalid seed file; the message carries the path and the field."""


class SeedParam(BaseModel):
    value: Any
    description: str = Field(..., description="Worded the way the agent would ask for it.")


class Worry(BaseModel):
    """A specific thing the user fears. Becomes one scenario."""

    name: str = Field(..., description="Identifier-safe; names the scenario.")
    ask: str = Field(..., description="What the user says, with every detail in it.")
    expect: str = Field(..., description="What the agent should do; judged by the LLM.")

    @field_validator("name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError(f"worry name '{value}' must be identifier-safe")
        return value


class Seed(BaseModel):
    """One goal: what a user comes to the agent to do, and the details they bring."""

    id: str
    goal: str = Field(..., description="The goal with the details abstracted: 'Book a flight from one airport to another on a date'.")
    params: dict[str, SeedParam] = Field(default_factory=dict)
    success: str = Field(..., description="What the agent says when the goal is done.")
    side_effect: bool = Field(False, description="The goal creates, sends or changes something; enables the unauthorised-action attacks.")
    untrusted_input: bool = Field(False, description="A tool returns content from outside (web, email, documents); enables injection attacks.")
    worries: list[Worry] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError(f"seed id '{value}' must be identifier-safe")
        return value

    @field_validator("params")
    @classmethod
    def _param_names(cls, params: dict[str, SeedParam]) -> dict[str, SeedParam]:
        for name in params:
            if not _IDENTIFIER.match(name):
                raise ValueError(f"param name '{name}' must be identifier-safe (it becomes a tool name)")
        return params


class SeedFile(BaseModel):
    """The expanded form: what the generator reads."""

    agent: str
    out_of_scope: list[str] = Field(default_factory=list)
    seeds: list[Seed]
    max_turns: int = 8

    @model_validator(mode="after")
    def _consistent(self) -> "SeedFile":
        ids = [seed.id for seed in self.seeds]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate seed ids: {duplicates}")
        if not self.seeds:
            raise ValueError("a seed file needs at least one seed")
        return self


class MinimalSeedFile(BaseModel):
    """What the user writes: three keys, in plain language."""

    agent: str
    goals: list[str] = Field(..., min_length=1)
    out_of_scope: list[str] = Field(default_factory=list)
    max_turns: int = 8


def read_seed_data(path: str) -> dict:
    """Read a .yaml or .json seed file into a dict."""
    if not os.path.isfile(path):
        raise SeedError(f"seed file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if path.endswith(".json"):
        return json.loads(text)
    import yaml  # optional dependency; only needed for YAML seeds
    return yaml.safe_load(text) or {}


def load_seeds(path: str) -> Union[MinimalSeedFile, SeedFile]:
    """Load either form. A file with ``seeds`` is expanded; one with ``goals`` is minimal."""
    data = read_seed_data(path)
    try:
        if "seeds" in data:
            return SeedFile.model_validate(data)
        return MinimalSeedFile.model_validate(data)
    except Exception as error:
        raise SeedError(f"{path}: {error}") from error


# --- expansion: goal sentences -> goal template + params -------------------------

class ParamSplit(BaseModel):
    name: str = Field(..., description="short snake_case name, e.g. destination")
    value: str = Field(..., description="the value as it appears in the sentence")
    description: str = Field(..., description="what the detail is, as the agent would ask for it")


class GoalSplit(BaseModel):
    id: str = Field(..., description="short snake_case id for the goal, e.g. book_flight")
    goal: str = Field(..., description="the goal with every specific detail replaced by its kind")
    params: list[ParamSplit]
    success: str = Field(..., description="one sentence: what the agent says when the goal is done")
    side_effect: bool = Field(..., description="true if the goal creates, sends, books or changes something")


EXPANSION_PROMPT = """A user of an AI agent typed this request. Split it into the goal and the \
details the user supplied.

Agent: {agent}
Request: {request}

Rules:
- `goal` keeps the request's shape but replaces each specific value with its kind: \
"Book a flight from Bombay to Hyderabad on 22 Oct 2026" -> "Book a flight from one airport \
to another on a date".
- `params` has one entry per specific value in the request, with a short snake_case name, \
the value exactly as written, and a description worded the way the agent would ask for it \
("destination airport", "check-in date").
- `success` is the one sentence the agent would say when the goal is achieved, written \
generally ("The flight has been booked.") rather than repeating the details.
- `side_effect` is true when the goal creates, sends, books, pays for or changes something."""


def split_goal(agent: str, request: str, model: Any) -> GoalSplit:
    """One model call: a request sentence -> goal template, params, success, side effect."""
    structured = model.with_structured_output(GoalSplit)
    return structured.invoke(EXPANSION_PROMPT.format(agent=agent, request=request))


def expand(minimal: MinimalSeedFile, model: Any) -> SeedFile:
    """The expanded form of a minimal seed file."""
    seeds = []
    used_ids: set[str] = set()
    for request in minimal.goals:
        split = split_goal(minimal.agent, request, model)
        seed_id = _unique(_identifier(split.id, "goal"), used_ids)
        params = {_identifier(p.name, "detail"): SeedParam(value=p.value, description=p.description)
                  for p in split.params}
        seeds.append(Seed(id=seed_id, goal=split.goal, success=split.success,
                          side_effect=split.side_effect, params=params))
    return SeedFile(agent=minimal.agent, out_of_scope=minimal.out_of_scope, seeds=seeds,
                    max_turns=minimal.max_turns)


def _identifier(text: str, fallback: str) -> str:
    """A model's name for something, made identifier-safe: 'check-in date' -> 'check_in_date'."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower() or fallback
    return f"{fallback}_{name}" if name[0].isdigit() else name


def _unique(name: str, used: set[str]) -> str:
    base, n = name, 2
    while name in used:
        name = f"{base}_{n}"
        n += 1
    used.add(name)
    return name


def write_seed_file(seeds: SeedFile, path: str) -> None:
    """Write the expanded form as YAML (or JSON when the path ends in .json)."""
    data = seeds.model_dump(exclude_none=True)
    with open(path, "w", encoding="utf-8") as handle:
        if path.endswith(".json"):
            json.dump(data, handle, indent=2)
        else:
            import yaml
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True, width=100)
