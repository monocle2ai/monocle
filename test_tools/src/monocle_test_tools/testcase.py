import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from opentelemetry.sdk.trace import ReadableSpan
from monocle_test_tools.schema import FactID, SpanType
from monocle_test_tools.trace_utils import get_input_from_span, get_output_from_span

if TYPE_CHECKING:
    from monocle_test_tools.fluent_api import TraceAssertion

# Turn spans are tagged "agentic.turn"; traces recorded by older Monocle
# versions tag the same span "agentic.request".
TURN_SPAN_TYPES = (SpanType.AGENTIC_REQUEST.value, "agentic.request")
INFERENCE_SPAN_TYPES = (SpanType.INFERENCE.value, "inference.framework")

def _keyed_entry(model: type[BaseModel], data: Any) -> Optional[Tuple[str, Any]]:
    """Split a ``{"<name>": <body>}`` mapping, or return None when data isn't one.

    A mapping that carries any of the model's own field names is the flat form
    (``{"name": ..., ...}``) and is left alone.
    """
    if not isinstance(data, dict) or not data or set(data) & set(model.model_fields):
        return None
    kind = model.__name__.lower()
    if len(data) > 1:
        raise ValueError(
            f'a {kind} must be a single "<name>": ... entry, got {sorted(data)}')
    return next(iter(data.items()))

class KeyedByName(BaseModel):
    """Base for models written in JSON as ``{"<name>": {<the other fields>}}``.

    The name is the JSON key, so a list of them reads as a list of named entries.
    The flat form (``{"name": ..., "input": ...}``) and a bare name string are
    also accepted on input; both serialize back out in the keyed form. Fields
    left unset are left out of the body, so a name-only entry is ``{"<name>": {}}``.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = Field(None, description="name")

    @model_validator(mode="before")
    @classmethod
    def _from_keyed_form(cls, data: Any) -> Any:
        """Turn ``{"<name>": {...}}`` (or a bare name) into the model's own fields."""
        if isinstance(data, str):
            return {"name": data}
        entry = _keyed_entry(cls, data)
        if entry is None:
            return data
        name, body = entry
        if body is not None and not isinstance(body, dict):
            raise ValueError(f'{cls.__name__.lower()} "{name}" must map to an object, '
                             f'got {type(body).__name__}')
        return {"name": name, **(body or {})}

    @model_serializer(mode="wrap")
    def _to_keyed_form(self, handler) -> dict:
        body = handler(self)
        name = body.pop("name", None)
        return {name: {key: value for key, value in body.items() if value is not None}}

class Agent(KeyedByName):
    """An agent to validate, written as ``{"<name>": {"input": ..., "output": ...}}``."""
    name: str = Field(None, description="agent name")
    input: Optional[str] = Field(None, description="agent input")
    output: Optional[str] = Field(None, description="agent output")

class Tool(KeyedByName):
    """A tool to validate, written as ``{"<name>": {"input": ..., "output": ..., "agent": ...}}``."""
    name: str = Field(None, description="tool name")
    input: Optional[str] = Field(None, description="tool input")
    output: Optional[str] = Field(None, description="tool output")
    agent: Optional[Agent] = Field(None, description="tool calling agent")

class Eval(BaseModel):
    """An eval to run, written as ``{"<name>": "<result>"}``.

    The eval name is the key and its expected result is the value, so the body is
    a single level. A category is carried in the key too, as
    ``{"<name>@<category>": "<result>"}`` -- so ``hallucination@llm`` is the
    ``hallucination`` eval in the ``llm`` category. The flat form
    (``{"name": ..., "category": ..., "result": ...}``) and a bare name string
    are also accepted on input.
    """
    model_config = ConfigDict(extra="forbid")

    name: Union[str, Path] = Field(None, description= " Eval name")
    category: Optional[str] = Field(None, description="Eval category")
    result: str = Field(None, description="Eval result")

    @staticmethod
    def _split_category(name: Any) -> Tuple[Any, Optional[str]]:
        """Split ``"<name>@<category>"`` into its parts.

        Only strings split, and only on the LAST ``@`` -- an eval name may
        contain one. A Path is a custom-template location, never a
        name@category pair, so it is returned whole. A trailing ``@`` names no
        category rather than an empty one.
        """
        if not isinstance(name, str) or "@" not in name:
            return name, None
        head, _, category = name.rpartition("@")
        if not head or not category:
            return name.rstrip("@") or name, None
        return head, category

    @model_validator(mode="before")
    @classmethod
    def _from_keyed_form(cls, data: Any) -> Any:
        """Turn ``{"<name>": <result>}`` (or a bare name) into the model's own fields."""
        if isinstance(data, (str, Path)):
            name, category = cls._split_category(data)
            return {"name": name, "category": category}
        entry = _keyed_entry(cls, data)
        if entry is None:
            return data
        name, result = entry
        if isinstance(result, dict):
            raise ValueError(f'eval "{name}" must map to a result value, not an object')
        name, category = cls._split_category(name)
        return {"name": name, "category": category, "result": result}

    @model_serializer(mode="wrap")
    def _to_keyed_form(self, handler) -> dict:
        body = handler(self)
        name, category = body.get("name"), body.get("category")
        return {f"{name}@{category}" if category else name: body.get("result")}

def _as_keyed_list(model: type[BaseModel], value: Any) -> Any:
    """Normalize the mapping form of a keyed-entry list field into a list.

    ``{"a": {...}, "b": {...}}`` becomes ``[{"a": {...}}, {"b": {...}}]`` so each
    item is a single keyed entry the item model already knows how to parse. A dict
    carrying the item model's own field names is instead a single flat entry and is
    just wrapped. Anything that is not a dict is left alone.

    ``_keyed_entry`` applies the same "carries field names" test but cannot be
    reused here: it raises on a dict with more than one key, which is exactly the
    mapping case.
    """
    if not isinstance(value, dict):
        return value
    if set(value) & set(model.model_fields):
        return [value]
    return [{name: body} for name, body in value.items()]

class FluentTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field("monocle_test", description="Name of the test case.")
    input: Optional[Union[Tuple[Any, ...], FactID]] = Field(None, description="Input prompt or data for the test case or fact_id/fact_name.")
    output: Optional[Union[str, list[str]]] = Field(None, description="expected output of the run as a whole, checked when no entity was selected")
    agents: Optional[list[Agent]] = Field([], description="agents to validate")
    tools: Optional[list[Tool]] = Field([], description="tools to validate")
    evals: Optional[list[Eval]] = Field([], description="evals to run")
    token_limit: Optional[int] = Field(None, description="Token limit")
    load_error: Optional[str] = Field(
        None,
        description="why this case's data could not be loaded; set by "
                    "setup_test_cases so an unloadable fact is reported as a "
                    "failing test rather than dropped from the suite")

    @model_validator(mode="before")
    @classmethod
    def _normalize_input_shapes(cls, data: Any) -> Any:
        """Accept the shapes parametrized tests are actually written in.

        Three normalizations, all input-only - serialization still emits the flat
        form, so a dump round-trips back through here unchanged:

        1. ``{"expected": {...}}`` has its contents lifted to the top level, so a
           test case can group its expectations without the model growing a nesting
           level it does not otherwise need.
        2. ``agents``/``tools``/``evals`` written as a mapping of name to body
           become the list of keyed entries the fields are declared as.
        3. A scalar ``input`` becomes a one-tuple, so ``"Book a flight"`` works
           where ``("Book a flight",)`` was required.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        expected = data.pop("expected", None)
        if isinstance(expected, dict):
            collisions = sorted(set(expected) & set(data))
            if collisions:
                raise ValueError(
                    f"{collisions} given both inside and outside 'expected'; "
                    "put each key in one place only")
            data.update(expected)
        elif expected is not None:
            raise ValueError(
                f"'expected' must be an object, got {type(expected).__name__}")

        for field, model in (("agents", Agent), ("tools", Tool), ("evals", Eval)):
            if field in data:
                data[field] = _as_keyed_list(model, data[field])

        value = data.get("input")
        if value is not None and not isinstance(value, (tuple, list, dict, FactID)):
            data["input"] = (value,)

        return data

    def validate_tools(self, asserter: "TraceAssertion") -> "FluentTestCase":
        """Assert on the given trace asserter that every tool of this test case was called.

        A tool's input and output are only checked when the test case sets them, so a
        name-only tool asserts nothing beyond the call having happened. The calling
        agent, when set, narrows the assertion to calls made by that agent.

        Every tool starts from ``asserter`` itself: ``called_tool`` returns a new
        asserter narrowed to the spans it matched, so carrying that one over to the
        next tool would look for it inside the previous tool's spans. Failures are
        collected on the asserter (they do not raise here) the way they are for a
        hand-written fluent chain.

        Args:
            asserter: The trace asserter holding the spans to assert on.

        Returns:
            This test case, so validations can be chained.
        """
        for tool in self.tools or []:
            scope = asserter.called_tool(
                tool.name, agent_name=tool.agent.name if tool.agent else None)
            if tool.input is not None:
                scope = scope.has_input(tool.input)
            if tool.output is not None:
                scope.has_output(tool.output)
        return self

    @classmethod
    def from_spans(cls, spans: Sequence[ReadableSpan], name: Optional[str] = None,
                   input: Optional[Union[Tuple[Any, ...], FactID]] = None,  # pylint: disable=redefined-builtin
                   evals: Optional[list[Union["Eval", dict]]] = None) -> "FluentTestCase":
        """Build a FluentTestCase from the spans of a recorded run.

        The spans are read the way the trace assertions read them, so the returned
        test case describes what the recorded run actually did: every agent that was
        invoked, every tool that was called (with its calling agent), the turn input,
        and the tokens the run consumed as the token limit.

        Identical repeats of an agent/tool call are collapsed, while calls of the
        same agent/tool with a different input or output are kept as separate
        entries. Spans are processed in start_time order, so agents and tools come
        out in call order regardless of the order the spans were loaded in.

        Evals cannot be derived from spans (a span records what happened, not the
        expected eval result), so they are only set when passed in.

        Args:
            spans: Spans of the recorded run, in any order.
            name: Test case name. Defaults to the trace's workflow name, and to the
                FluentTestCase default when the spans carry no workflow span.
            input: Test case input. Defaults to the inputs of the turn spans (one
                entry per turn), falling back to the first agent invocation's input.
                Pass a FactID to point the test case at a stored fact instead.
            evals: Evals to run, as Eval objects or dicts.

        Returns:
            A FluentTestCase describing the recorded run.
        """
        ordered_spans = sorted(spans or [], key=lambda span: getattr(span, "start_time", 0) or 0)
        workflow_name = _first_workflow_name(ordered_spans)
        turn_inputs = [text for text in
                       (_as_text(get_input_from_span(span)) for span in ordered_spans
                        if _span_type(span) in TURN_SPAN_TYPES) if text]
        agents = _dedupe([_agent_from_span(span) for span in ordered_spans
                          if _span_type(span) == SpanType.AGENTIC_INVOCATION.value
                          and span.attributes.get("entity.1.name")])
        tools = _dedupe([_tool_from_span(span) for span in ordered_spans
                         if _span_type(span) == SpanType.TOOL_INVOCATION.value
                         and span.attributes.get("entity.1.name")])
        if not turn_inputs:
            # No turn span among the spans (a partial trace, or a framework that
            # emits none) - fall back to what the first agent was asked to do.
            turn_inputs = [agent.input for agent in agents[:1] if agent.input]

        return cls(
            **({"name": name or workflow_name} if (name or workflow_name) else {}),
            input=input if input is not None else (tuple(turn_inputs) or None),
            agents=agents,
            tools=tools,
            evals=[ev if isinstance(ev, Eval) else Eval.model_validate(ev) for ev in evals or []],
            token_limit=_total_tokens(ordered_spans) or None,
        )


def load_test_cases_from_json(path: Union[str, Path]) -> list["FluentTestCase"]:
    """Load a JSON array of test cases from a file.

    Each element is parsed by FluentTestCase itself, so a committed file may be
    written in any shape a parametrize literal can use -- the ``expected``
    wrapper, evals as a name-to-result mapping, a scalar input. The model also
    serializes back into a shape it accepts, so a discovered set can be dumped,
    committed, and reloaded later: that round trip is the point of this, turning
    "what the evals said today" into a golden dataset with no network call.

    Args:
        path: Path to a .json file holding an array of test cases.

    Returns:
        The test cases, in file order.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not valid JSON, does not hold an array, or
            holds an element that is not a valid test case. Every message names
            the path, and an invalid element names its index -- a typo in case 7
            of 40 should say 7.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Test case file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Test case file is not valid JSON: {path} - {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Test case file must hold an array of test cases, got "
            f"{type(data).__name__}: {path}")

    test_cases = []
    for index, entry in enumerate(data):
        try:
            test_cases.append(FluentTestCase.model_validate(entry))
        except ValueError as exc:
            raise ValueError(
                f"test case {index} in {path} is not valid: {exc}") from exc
    return test_cases


def _span_type(span: ReadableSpan) -> str:
    return (span.attributes or {}).get("span.type", "")

def _first_workflow_name(spans: Sequence[ReadableSpan]) -> Optional[str]:
    """Workflow name of the run, carried on the workflow span as entity.1.name."""
    for span in spans:
        if _span_type(span) == "workflow":
            name = span.attributes.get("entity.1.name")
            if name:
                return name
    return None

def _agent_from_span(span: ReadableSpan) -> "Agent":
    return Agent(name=span.attributes.get("entity.1.name"),
                 input=_as_text(get_input_from_span(span)),
                 output=_as_text(get_output_from_span(span)))

def _tool_from_span(span: ReadableSpan) -> "Tool":
    calling_agent = span.attributes.get("entity.2.name")
    return Tool(name=span.attributes.get("entity.1.name"),
                input=_as_text(get_input_from_span(span)),
                output=_as_text(get_output_from_span(span)),
                agent=Agent(name=calling_agent) if calling_agent else None)

def _total_tokens(spans: Sequence[ReadableSpan]) -> int:
    """Tokens consumed across the run, from the metadata event of inference spans."""
    total = 0
    for span in spans:
        if _span_type(span) in INFERENCE_SPAN_TYPES:
            for event in getattr(span, "events", None) or []:
                if event.name == "metadata":
                    total += event.attributes.get("total_tokens", 0) or 0
    return total

def _as_text(value: Any) -> Optional[str]:
    """Span event values are usually strings but can be lists/dicts - keep them as text."""
    if value is None or isinstance(value, str):
        return value
    return str(value)

def _dedupe(items: list[Any]) -> list[Any]:
    """Drop identical repeats, preserving first-seen (call) order."""
    deduped, seen = [], set()
    for item in items:
        key = item.model_dump_json()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped

test_example1 = {
    "input": "Book a flight from SFO to LAX for tomorrow",
    "agents": [
        {"supervisor": {}},
        {"adk_book_hotel": {"output": "Hotel booked"}}
    ],
    "tools": [
        {"book_flight": {"output": "confirmed", "agent": {"adk_book_flight": {}}}}
    ],
}

test_example2 = {
    "input": {"fact_id": "12345"},
    "agents": [
        {"adk_book_fligh": {}}
    ],
    "evals": [
        {"hallucinations": "major_hallucination"}
    ],
}

# Create a combined API for run_agent and with_trace_source
# Create a test example that uses the expected value via FluentTestCase
# Create iterator of FluentTestCase
