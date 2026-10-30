"""Unit tests for the OpenAI Agents SDK runner.

The agents run through the real ``agents.Runner`` against a scripted fake
model, so they need no API key, model provider, or network access.
"""
import pytest
from agents import Agent, function_tool
from agents.models.interface import Model
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from monocle_test_tools import MonocleValidator, MultiTurnTestCase
from monocle_test_tools.runner.runner import get_agent_runner, AgentTypes
from monocle_test_tools.runner.openai_runner import OpenAIRunner


def _message(text):
    return ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _tool_call(name, arguments):
    return ResponseFunctionToolCall(
        id="fc_1",
        call_id="call_1",
        type="function_call",
        name=name,
        arguments=arguments,
    )


class ScriptedModel(Model):
    """Returns the scripted outputs in order and records the input of each call."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.inputs = []

    async def get_response(self, system_instructions, input, *args, **kwargs):
        self.inputs.append(input)
        return ModelResponse(output=[self._outputs.pop(0)], usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def _user_messages(model_input):
    """User message texts the model received on one call."""
    if isinstance(model_input, str):
        return [model_input]
    return [item["content"] for item in model_input if item.get("role") == "user"]


@pytest.fixture
def weather_agent():
    tool_calls = []

    @function_tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        tool_calls.append(city)
        return f"Sunny in {city}"

    model = ScriptedModel([
        _tool_call("get_weather", '{"city": "Paris"}'),
        _message("It is sunny in Paris."),
    ])
    agent = Agent(name="weather_agent", instructions="Answer weather questions.",
                  model=model, tools=[get_weather])
    return agent, model, tool_calls


def _chat_agent(replies):
    model = ScriptedModel([_message(reply) for reply in replies])
    return Agent(name="chat_agent", instructions="Chat.", model=model), model


def test_agent_type_mapping():
    assert AgentTypes.OPENAI.value == "openai"
    assert isinstance(get_agent_runner("openai"), OpenAIRunner)


@pytest.mark.asyncio
async def test_run_agent_async_returns_final_output(weather_agent):
    agent, model, tool_calls = weather_agent
    result = await OpenAIRunner().run_agent_async(agent, "What is the weather in Paris?")
    assert result == "It is sunny in Paris."
    assert tool_calls == ["Paris"]
    assert len(model.inputs) == 2


def test_run_agent_returns_final_output(weather_agent):
    agent, model, tool_calls = weather_agent
    result = OpenAIRunner().run_agent(agent, "What is the weather in Paris?")
    assert result == "It is sunny in Paris."
    assert tool_calls == ["Paris"]


def test_run_agent_keeps_conversation_across_turns():
    agent, model = _chat_agent(["Hello Alice.", "Your name is Alice."])
    runner = OpenAIRunner()
    runner.run_agent(agent, "My name is Alice.", session_id="s1")
    result = runner.run_agent(agent, "What is my name?", session_id="s1")
    assert result == "Your name is Alice."
    assert _user_messages(model.inputs[1]) == ["My name is Alice.", "What is my name?"]


@pytest.mark.asyncio
async def test_session_id_keeps_conversation_across_turns():
    agent, model = _chat_agent(["Hello Alice.", "Your name is Alice."])
    runner = OpenAIRunner()
    await runner.run_agent_async(agent, "My name is Alice.", session_id="s1")
    result = await runner.run_agent_async(agent, "What is my name?", session_id="s1")
    assert result == "Your name is Alice."
    assert _user_messages(model.inputs[1]) == ["My name is Alice.", "What is my name?"]


@pytest.mark.asyncio
async def test_without_session_id_turns_are_independent():
    agent, model = _chat_agent(["Hello Alice.", "I don't know."])
    runner = OpenAIRunner()
    await runner.run_agent_async(agent, "My name is Alice.")
    await runner.run_agent_async(agent, "What is my name?")
    assert _user_messages(model.inputs[1]) == ["What is my name?"]


@pytest.mark.asyncio
async def test_sessions_are_isolated_by_id():
    agent, model = _chat_agent(["Hello Alice.", "Hello Bob."])
    runner = OpenAIRunner()
    await runner.run_agent_async(agent, "My name is Alice.", session_id="s1")
    await runner.run_agent_async(agent, "My name is Bob.", session_id="s2")
    assert _user_messages(model.inputs[1]) == ["My name is Bob."]


@pytest.mark.asyncio
async def test_end_session_drops_conversation():
    agent, model = _chat_agent(["Hello Alice.", "I don't know."])
    runner = OpenAIRunner()
    await runner.run_agent_async(agent, "My name is Alice.", session_id="s1")
    await runner.end_session("s1")
    await runner.run_agent_async(agent, "What is my name?", session_id="s1")
    assert _user_messages(model.inputs[1]) == ["What is my name?"]


@pytest.mark.asyncio
async def test_end_session_unknown_id_is_noop():
    await OpenAIRunner().end_session("missing")


@pytest.mark.asyncio
async def test_validator_runs_openai_agent(weather_agent):
    agent, model, tool_calls = weather_agent
    validator = MonocleValidator()
    try:
        result = await validator.run_agent_async(
            agent, AgentTypes.OPENAI, "What is the weather in Paris?", session_id="s1"
        )
    finally:
        validator.cleanup()
    assert result == "It is sunny in Paris."
    assert tool_calls == ["Paris"]


@pytest.mark.asyncio
async def test_validator_runs_multi_turn_openai_agent():
    agent, model = _chat_agent(["Hello Alice.", "Your name is Alice."])
    multi_turn_case = MultiTurnTestCase(
        session_id="s1",
        turns=[{"test_input": ["My name is Alice."]}, {"test_input": ["What is my name?"]}],
    )
    validator = MonocleValidator()
    try:
        _, results, _ = await validator.run_multi_turn_agent_async(agent, AgentTypes.OPENAI, multi_turn_case)
    finally:
        validator.cleanup()
    assert results == ["Hello Alice.", "Your name is Alice."]
    assert _user_messages(model.inputs[1]) == ["My name is Alice.", "What is my name?"]
