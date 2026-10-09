"""Template for the LangChain test agent: prompt assembly and agent construction.

Copy this module as a starting point if a scenario needs a differently-shaped tester.
The driver only requires that ``build_test_agent`` return an object with an
``ainvoke({"messages": [...]})`` coroutine and the live param-call log.
"""
from typing import Any

from monocle_test_tools.harness.param_tools import build_param_tools
from monocle_test_tools.harness.preamble import PREAMBLE
from monocle_test_tools.harness.scenario import ScenarioTestCase

TEST_AGENT_NAME = "monocle_test_agent"


def build_system_prompt(case: ScenarioTestCase, target_description: str) -> str:
    """Assemble preamble + target description + scenario + persona + criteria + params.

    Values of non-initial params are deliberately absent: only their names and
    descriptions appear, so the test agent cannot state a withheld detail without
    calling its tool first.
    """
    lines = [
        PREAMBLE,
        "",
        "[TARGET AGENT]",
        target_description,
        "",
        "[SCENARIO]",
        case.scenario,
        "",
        "[YOUR PERSONA]",
        case.persona,
        "",
        "[SUCCESS CRITERIA]",
        case.success_criteria,
    ]
    if case.initial_params:
        lines += ["", "[WHAT YOU KNOW UP FRONT]"]
        lines += [f"  {param.name}: {param.value}   ({param.description})"
                  for param in case.initial_params]
    if case.on_request_params:
        lines += ["",
                  "[DETAILS AVAILABLE ON REQUEST -- call the matching tool when asked]"]
        lines += [f"  {param.name} -- {param.description}"
                  for param in case.on_request_params]
    return "\n".join(lines)


def build_test_agent(case: ScenarioTestCase, target_description: str,
                     model: Any) -> tuple[Any, list[str]]:
    """Build the test agent.

    Returns ``(agent, param_calls)`` where ``param_calls`` is the live list the param
    tools append to when invoked.
    """
    from langchain.agents import create_agent

    param_tools, param_calls = build_param_tools(case.params)
    agent = create_agent(
        model=model,
        tools=param_tools,
        system_prompt=build_system_prompt(case, target_description),
        name=TEST_AGENT_NAME,
    )
    return agent, param_calls
