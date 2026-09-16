"""Unit tests for the keyed JSON shapes: Agent/Tool as {"<name>": {...}}, Eval as {"<name>": <result>}."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from monocle_test_tools.testcase import Agent, Eval, FluentTestCase, Tool


def test_agent_serializes_name_as_the_key():
    agent = Agent(name="supervisor", input="book a flight", output="booked")

    assert agent.model_dump() == {
        "supervisor": {"input": "book a flight", "output": "booked"}
    }


def test_agent_serializes_name_only_agent_as_empty_body():
    assert Agent(name="supervisor").model_dump() == {"supervisor": {}}


def test_agent_json_uses_the_keyed_shape():
    agent = Agent(name="supervisor", input="in", output="out")

    assert json.loads(agent.model_dump_json()) == {
        "supervisor": {"input": "in", "output": "out"}
    }


def test_agent_deserializes_from_the_keyed_shape():
    agent = Agent.model_validate({"supervisor": {"input": "in", "output": "out"}})

    assert (agent.name, agent.input, agent.output) == ("supervisor", "in", "out")


def test_agent_deserializes_from_the_keyed_shape_with_empty_body():
    agent = Agent.model_validate({"supervisor": {}})

    assert (agent.name, agent.input, agent.output) == ("supervisor", None, None)


def test_agent_deserializes_from_a_bare_name():
    agent = Agent.model_validate("supervisor")

    assert (agent.name, agent.input, agent.output) == ("supervisor", None, None)


def test_agent_still_deserializes_from_the_flat_shape():
    agent = Agent.model_validate({"name": "supervisor", "input": "in"})

    assert (agent.name, agent.input, agent.output) == ("supervisor", "in", None)


def test_agent_round_trips_through_the_keyed_shape():
    agent = Agent(name="supervisor", input="in", output="out")

    assert Agent.model_validate(agent.model_dump()) == agent


def test_agent_rejects_a_multi_key_mapping():
    with pytest.raises(ValidationError):
        Agent.model_validate({"agent_a": {}, "agent_b": {}})


def test_agent_rejects_an_unknown_key_in_the_body():
    with pytest.raises(ValidationError):
        Agent.model_validate({"supervisor": {"prompt": "in"}})


def test_tool_serializes_name_as_the_key():
    tool = Tool(name="book_flight", input="{'to': 'LAX'}", output="confirmed")

    assert tool.model_dump() == {
        "book_flight": {"input": "{'to': 'LAX'}", "output": "confirmed"}
    }


def test_tool_serializes_name_only_tool_as_empty_body():
    assert Tool(name="book_flight").model_dump() == {"book_flight": {}}


def test_tool_serializes_its_calling_agent_in_the_keyed_shape():
    tool = Tool(name="book_flight", output="confirmed", agent=Agent(name="flight_agent"))

    assert tool.model_dump() == {
        "book_flight": {"output": "confirmed", "agent": {"flight_agent": {}}}
    }


def test_tool_json_uses_the_keyed_shape():
    tool = Tool(name="book_flight", input="in", agent=Agent(name="flight_agent"))

    assert json.loads(tool.model_dump_json()) == {
        "book_flight": {"input": "in", "agent": {"flight_agent": {}}}
    }


def test_tool_deserializes_from_the_keyed_shape():
    tool = Tool.model_validate({"book_flight": {"input": "in", "output": "out"}})

    assert (tool.name, tool.input, tool.output) == ("book_flight", "in", "out")


def test_tool_deserializes_from_the_keyed_shape_with_empty_body():
    tool = Tool.model_validate({"book_flight": {}})

    assert (tool.name, tool.input, tool.output, tool.agent) == ("book_flight", None, None, None)


def test_tool_deserializes_from_a_bare_name():
    assert Tool.model_validate("book_flight").name == "book_flight"


def test_tool_deserializes_its_calling_agent_from_the_keyed_shape():
    tool = Tool.model_validate({"book_flight": {"agent": {"flight_agent": {}}}})

    assert tool.agent == Agent(name="flight_agent")


def test_tool_still_deserializes_from_the_flat_shape():
    tool = Tool.model_validate({"name": "book_flight", "input": "in",
                                "agent": {"flight_agent": {}}})

    assert (tool.name, tool.input, tool.agent) == ("book_flight", "in", Agent(name="flight_agent"))


def test_tool_round_trips_through_the_keyed_shape():
    tool = Tool(name="book_flight", input="in", output="out", agent=Agent(name="flight_agent"))

    assert Tool.model_validate(tool.model_dump()) == tool


def test_tool_rejects_a_multi_key_mapping():
    with pytest.raises(ValidationError):
        Tool.model_validate({"tool_a": {}, "tool_b": {}})


def test_tool_rejects_an_unknown_key_in_the_body():
    with pytest.raises(ValidationError):
        Tool.model_validate({"book_flight": {"args": "in"}})


def test_eval_serializes_name_as_the_key_and_result_as_the_value():
    assert Eval(name="hallucinations", result="none").model_dump() == {"hallucinations": "none"}


def test_eval_serializes_a_result_less_eval_with_a_null_value():
    assert Eval(name="hallucinations").model_dump() == {"hallucinations": None}


def test_eval_json_uses_the_keyed_shape():
    eval_ = Eval(name="hallucinations", result="major_hallucination")

    assert json.loads(eval_.model_dump_json()) == {"hallucinations": "major_hallucination"}


def test_eval_json_renders_a_template_path_name_as_the_key():
    eval_ = Eval(name=Path("evals") / "tone.json", result="pass")

    assert json.loads(eval_.model_dump_json()) == {"evals/tone.json": "pass"}


def test_eval_deserializes_from_the_keyed_shape():
    eval_ = Eval.model_validate({"hallucinations": "major_hallucination"})

    assert (eval_.name, eval_.result) == ("hallucinations", "major_hallucination")


def test_eval_deserializes_from_a_bare_name():
    eval_ = Eval.model_validate("hallucinations")

    assert (eval_.name, eval_.result) == ("hallucinations", None)


def test_eval_still_deserializes_from_the_flat_shape():
    eval_ = Eval.model_validate({"name": "hallucinations", "result": "none"})

    assert (eval_.name, eval_.result) == ("hallucinations", "none")


def test_eval_round_trips_through_the_keyed_shape():
    eval_ = Eval(name="hallucinations", result="none")

    assert Eval.model_validate(eval_.model_dump()) == eval_


def test_eval_rejects_a_multi_key_mapping():
    with pytest.raises(ValidationError):
        Eval.model_validate({"hallucinations": "none", "tone": "pass"})


def test_eval_rejects_an_object_as_its_result():
    with pytest.raises(ValidationError):
        Eval.model_validate({"hallucinations": {"result": "none"}})


def test_fluent_test_case_round_trips_agents_in_the_keyed_shape():
    test_case = FluentTestCase.model_validate({
        "input": ("Book a flight to LAX",),
        "agents": [{"supervisor": {"input": "Book a flight to LAX"}},
                   {"flight_agent": {}}],
    })

    assert [(a.name, a.input) for a in test_case.agents] == [
        ("supervisor", "Book a flight to LAX"), ("flight_agent", None)]
    assert test_case.model_dump()["agents"] == [
        {"supervisor": {"input": "Book a flight to LAX"}}, {"flight_agent": {}}]


def test_fluent_test_case_round_trips_tools_in_the_keyed_shape():
    test_case = FluentTestCase.model_validate({
        "tools": [{"book_flight": {"output": "confirmed",
                                   "agent": {"flight_agent": {}}}},
                  {"book_hotel": {}}],
    })

    assert [(t.name, t.output, t.agent) for t in test_case.tools] == [
        ("book_flight", "confirmed", Agent(name="flight_agent")),
        ("book_hotel", None, None)]
    assert test_case.model_dump()["tools"] == [
        {"book_flight": {"output": "confirmed", "agent": {"flight_agent": {}}}},
        {"book_hotel": {}}]


def test_fluent_test_case_round_trips_evals_in_the_keyed_shape():
    test_case = FluentTestCase.model_validate({
        "evals": [{"hallucinations": "major_hallucination"}, {"tone": "pass"}],
    })

    assert [(e.name, e.result) for e in test_case.evals] == [
        ("hallucinations", "major_hallucination"), ("tone", "pass")]
    assert test_case.model_dump()["evals"] == [
        {"hallucinations": "major_hallucination"}, {"tone": "pass"}]
