"""Tests for turning scenario params into LangChain tools. No LLM involved."""
from monocle_test_tools.harness.param_tools import ParamCall, build_param_tools
from monocle_test_tools.harness.scenario import ScenarioParam

PARAMS = [
    ScenarioParam(name="destination", value="Seattle",
                  description="destination airport"),
    ScenarioParam(name="date", value="22nd October 2026", description="travel date"),
]


def test_one_tool_per_param_named_after_the_param():
    tools, _calls = build_param_tools(PARAMS)
    assert [tool.name for tool in tools] == ["destination", "date"]


def test_tool_description_carries_the_param_description():
    tools, _calls = build_param_tools(PARAMS)
    assert "destination airport" in tools[0].description


def test_tool_returns_the_param_value():
    tools, _calls = build_param_tools(PARAMS)
    assert tools[0].invoke({}) == "Seattle"


def test_each_tool_returns_its_own_value():
    """Guards against the classic closure-over-loop-variable bug."""
    tools, _calls = build_param_tools(PARAMS)
    assert [tool.invoke({}) for tool in tools] == ["Seattle", "22nd October 2026"]


def test_invocation_is_recorded_in_the_call_log_with_its_value():
    """The log carries the returned value too, so a report can show what was fetched."""
    tools, calls = build_param_tools(PARAMS)
    assert calls == []
    tools[1].invoke({})
    assert calls == [ParamCall(name="date", value="22nd October 2026")]


def test_call_log_records_every_call_in_order():
    tools, calls = build_param_tools(PARAMS)
    tools[0].invoke({})
    tools[1].invoke({})
    assert [c.name for c in calls] == ["destination", "date"]
    assert [c.value for c in calls] == ["Seattle", "22nd October 2026"]


def test_non_string_values_are_stringified():
    tools, _calls = build_param_tools(
        [ScenarioParam(name="nights", value=4, description="number of nights")])
    assert tools[0].invoke({}) == "4"


def test_empty_params_yields_no_tools():
    tools, calls = build_param_tools([])
    assert tools == []
    assert calls == []
