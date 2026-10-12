"""Scenario params as zero-arg LangChain tools.

One tool per param. The test agent calls a tool when the target agent asks for that
detail; the value of a withheld param never appears in the prompt, so the tool call is
the only way the test agent can learn it.
"""
from typing import Any

from pydantic import BaseModel, Field

from monocle_test_tools.harness.scenario import ScenarioParam


class ParamCall(BaseModel):
    """One param tool invocation: which detail was fetched, and what it returned."""

    name: str = Field(..., description="Param (and tool) name.")
    value: str = Field(..., description="Value the tool returned.")


def build_param_tools(params: list[ScenarioParam]) -> tuple[list[Any], list[ParamCall]]:
    """Build one zero-arg tool per param.

    Returns ``(tools, calls)``. ``calls`` is a live list that each tool appends a
    :class:`ParamCall` to when invoked, recording which detail the test agent had to ask
    for and what it got back. The driver clears it before a turn and snapshots it after.
    """
    calls: list[ParamCall] = []
    return [_build_param_tool(param, calls) for param in params], calls


def _build_param_tool(param: ScenarioParam, calls: list[ParamCall]) -> Any:
    """Build the tool for one param.

    Separate function rather than an inline closure in the loop: binding ``param`` as
    an argument is what keeps each tool returning its own value instead of every tool
    closing over the last loop variable.
    """
    from langchain_core.tools import StructuredTool

    def fetch_param() -> str:
        value = str(param.value)
        calls.append(ParamCall(name=param.name, value=value))
        return value

    return StructuredTool.from_function(
        func=fetch_param,
        name=param.name,
        description=(
            f"Returns the {param.description}. Call this only when the target agent "
            f"asks you for the {param.description}."
        ),
    )
