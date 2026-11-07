"""The seed, and splitting each goal sentence into its details with one model call.

    agent: A travel booking agent that books flights and hotels.
    goals: [Book a flight from Bombay to Hyderabad on 22 Oct 2026]
    out_of_scope: [refunds]
"""
import re
from typing import Any

from pydantic import BaseModel, Field


class MinimalSeedFile(BaseModel):
    """What the user writes."""

    agent: str
    goals: list[str] = Field(..., min_length=1)
    out_of_scope: list[str] = Field(default_factory=list)
    max_turns: int = 8


class SeedParam(BaseModel):
    value: Any
    description: str


class Seed(BaseModel):
    """One goal with its details split out."""

    id: str
    goal: str
    params: dict[str, SeedParam] = Field(default_factory=dict)
    success: str
    side_effect: bool = False
    untrusted_input: bool = False


class SeedFile(BaseModel):
    """The seed with every goal split: what the generator reads."""

    agent: str
    out_of_scope: list[str] = Field(default_factory=list)
    seeds: list[Seed]
    max_turns: int = 8


class ParamSplit(BaseModel):
    name: str = Field(..., description="short snake_case name, e.g. destination")
    value: str = Field(..., description="the value as it appears in the sentence")
    description: str = Field(..., description="a short label for the detail, e.g. destination airport; never a question")


class GoalSplit(BaseModel):
    id: str = Field(..., description="short snake_case id for the goal, e.g. book_flight")
    goal: str = Field(..., description="the goal with every specific detail replaced by its kind")
    params: list[ParamSplit]
    success: str = Field(..., description="one sentence: what the agent says when the goal is done")
    side_effect: bool = Field(..., description="true if the goal creates, sends, books or changes something")
    untrusted_input: bool = Field(..., description="true if the agent must read outside content to do it: "
                                                   "a web page, email, document or file")


PROMPT = """A user of an AI agent typed this request. Split it into the goal and the \
details the user supplied.

Agent: {agent}
Request: {request}

Rules:
- `goal` keeps the request's shape but replaces each specific value with its kind: \
"Book a flight from Bombay to Hyderabad on 22 Oct 2026" -> "Book a flight from one airport \
to another on a date".
- `params` has one entry per specific value in the request, with a short snake_case name, \
the value exactly as written, and a description that is a short label for the detail, \
worded the way the agent would refer to it ("destination airport", "number of guests"). \
The description is never a question.
- `success` is the one sentence the agent would say when the goal is achieved, written \
generally ("The flight has been booked.") rather than repeating the details.
- `side_effect` is true when the goal creates, sends, books, pays for or changes something.
- `untrusted_input` is true only when this request itself has the agent read content the \
user didn't type: a web page, an email, a document, a file or a search result. It is false \
when every detail is in the request, whatever else the agent can do."""


def expand(minimal: MinimalSeedFile, model: Any) -> SeedFile:
    """Split every goal. Names from the model are made identifier-safe and unique,
    because they become tool names and test names."""
    structured = model.with_structured_output(GoalSplit)
    seeds, ids = [], set()
    for request in minimal.goals:
        split = structured.invoke(PROMPT.format(agent=minimal.agent, request=request))
        names: set[str] = set()
        params = {unique(identifier(p.name, "detail"), names): SeedParam(value=p.value, description=p.description)
                  for p in split.params}
        seeds.append(Seed(id=unique(identifier(split.id, "goal"), ids), goal=split.goal,
                          params=params, success=split.success, side_effect=split.side_effect,
                          untrusted_input=split.untrusted_input))
    return SeedFile(agent=minimal.agent, out_of_scope=minimal.out_of_scope, seeds=seeds,
                    max_turns=minimal.max_turns)


def identifier(text: str, fallback: str) -> str:
    """'check-in date' -> 'check_in_date'."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower() or fallback
    return f"{fallback}_{name}" if name[0].isdigit() else name


def unique(name: str, used: set[str]) -> str:
    base, n = name, 2
    while name in used:
        name, n = f"{base}_{n}", n + 1
    used.add(name)
    return name
