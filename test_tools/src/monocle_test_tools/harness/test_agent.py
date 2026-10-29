"""The simulated user: an LLM agent with one zero-argument tool per detail of the request.

A held-back detail's value is never in the prompt, only its name, so the simulated user
can't state it until the agent asks and it calls the tool.
"""
from typing import Any

from pydantic import BaseModel

from monocle_test_tools.harness.scenario import ScenarioParam, ScenarioTestCase

PREAMBLE = """You are a test agent. Your job is to test another AI agent (the "target \
agent") by conversing with it as a realistic end user. You are NOT an assistant and you \
must never help the target agent -- you are the customer.

Rules:
- Stay in your persona for every message, including the first.
- Call whatever tools you need first. Then send the message itself and nothing more: \
no commentary, no stage directions, no meta-talk about testing. Calling a tool is not \
commentary -- it is how you learn a detail you do not know.
- You have a tool for each detail of your request. Call a tool ONLY when the target \
agent asks you for that detail. Do NOT reveal a detail from the on-request list before \
the target agent asks for it. That restriction covers ONLY the on-request list -- the \
details you already know are yours to state freely, and you should open with them.
- Keep going until the success criteria are met, and adjust your next message to what \
the target agent just said.
- If the target agent goes off track, push it back toward the scenario, in persona.
- You only ever write the customer's side. Never take the target agent's role: do not \
ask it the questions it should be asking you, do not summarise the request back to it, \
do not offer help. If it asks for something you were not given, answer as a real \
customer would, for example that you have no preference."""

# Naming this marker in the prompt makes the model write the target agent's lines itself.
TARGET_REPLIED = "The target agent replied:\n"


def from_target(reply: Any) -> str:
    """The target agent's reply as the simulated user receives it, marked so it isn't answered in the agent's role."""
    return TARGET_REPLIED + str(reply)


class ParamCall(BaseModel):
    name: str
    value: str


def build_system_prompt(case: ScenarioTestCase, target_description: str) -> str:
    lines = [PREAMBLE, "", "[TARGET AGENT]", target_description, "", "[SCENARIO]", case.scenario, "",
             "[YOUR PERSONA]", case.persona, ""]
    if case.success_criteria:
        lines += ["[SUCCESS CRITERIA]", case.success_criteria]
    else:
        lines += ["[HOW TO CONTINUE]", "There is no checklist. Keep the conversation going the way this user "
                  "naturally would: follow up on what the agent says, ask related things, change your mind."]
    if case.initial_params:
        lines += ["", "[WHAT YOU KNOW UP FRONT]"]
        lines += [f"  {p.name}: {p.value}   ({p.description})" for p in case.initial_params]
    if case.on_request_params:
        lines += ["", "[DETAILS AVAILABLE ON REQUEST -- call the matching tool when asked]"]
        lines += [f"  {p.name} -- {p.description}" for p in case.on_request_params]
    return "\n".join(lines)


def build_param_tools(params: list[ScenarioParam]) -> tuple[list[Any], list[ParamCall]]:
    """One tool per param, and the live list each tool records its call in."""
    from langchain_core.tools import StructuredTool

    calls: list[ParamCall] = []

    def tool(param: ScenarioParam) -> Any:     # a function, so each tool keeps its own param
        def fetch() -> str:
            calls.append(ParamCall(name=param.name, value=str(param.value)))
            return str(param.value)
        return StructuredTool.from_function(
            func=fetch, name=param.name,
            description=f"Returns the {param.description}. Call this only when the target agent "
                        f"asks you for the {param.description}.")

    return [tool(p) for p in params], calls


def build_test_agent(case: ScenarioTestCase, target_description: str, model: Any) -> tuple[Any, list[ParamCall]]:
    from langchain.agents import create_agent

    tools, calls = build_param_tools(case.params)
    agent = create_agent(model=model, tools=tools, name="monocle_test_agent",
                         system_prompt=build_system_prompt(case, target_description))
    return agent, calls
