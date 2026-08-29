import logging
import os
import tempfile
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace import setup_monocle_telemetry
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.terminal import TerminalTool
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"


@pytest.fixture(scope="function")
def setup():
    try:
        memory_exporter = InMemorySpanExporter()
        custom_exporter = CustomConsoleSpanExporter()
        span_processors = [SimpleSpanProcessor(memory_exporter), SimpleSpanProcessor(custom_exporter)]
        instrumentor = setup_monocle_telemetry(
            workflow_name="openhands_agent_1",
            span_processors=span_processors
        )
        yield memory_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.fixture(scope="function")
def workspace():
    # OpenHands tools run against a real workspace directory.
    with tempfile.TemporaryDirectory() as workspace_dir:
        yield workspace_dir


def build_conversation(workspace_dir: str, api_key: str = None) -> Conversation:
    llm = LLM(
        model=MODEL,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        usage_id="openhands_test_llm",
    )
    agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name)])
    return Conversation(agent=agent, workspace=workspace_dir)


def test_single_agent(setup, workspace):
    conversation = build_conversation(workspace)
    conversation.send_message(
        "Use the terminal to run `echo OKAHU_TEST`. Then finish."
    )
    conversation.run()
    verify_spans(setup, str(conversation.state.id))


@pytest.mark.asyncio
async def test_async_multi_turn_shares_session(setup, workspace):
    """Every run of the same conversation is anchored to the conversation id, so a
    multi-turn session produces one turn span per run under a single session scope."""
    conversation = build_conversation(workspace)
    conversation.send_message("Use the terminal to run `echo FIRST_TURN`. Then finish.")
    await conversation.arun()
    conversation.send_message("Use the terminal to run `echo SECOND_TURN`. Then finish.")
    await conversation.arun()

    time.sleep(2)
    session_id = str(conversation.state.id)
    turn_spans = [
        span for span in setup.get_finished_spans()
        if span.attributes.get("span.type") == "agentic.turn"
    ]
    assert len(turn_spans) >= 2, f"Expected a turn span per run, got {len(turn_spans)}"
    for span in turn_spans:
        assert span.attributes["entity.1.type"] == "agent.openhands"
        assert span.attributes.get("scope.agentic.session") == session_id, \
            f"Expected session {session_id}, got {span.attributes.get('scope.agentic.session')}"


def test_invalid_api_key_error_code_in_span(setup, workspace):
    """Test that passing an invalid API key results in error_code in the span."""
    conversation = build_conversation(workspace, api_key="INVALID_API_KEY")
    conversation.send_message("Use the terminal to run `echo OKAHU_TEST`. Then finish.")
    with pytest.raises(Exception):
        conversation.run()

    time.sleep(2)
    found_turn = found_invocation = False
    for span in setup.get_finished_spans():
        span_attributes = span.attributes
        if span_attributes.get("span.type") in ("agentic.turn", "agentic.invocation"):
            span_input, span_output = span.events
            assert "error_code" in span_output.attributes
            if span_attributes["span.type"] == "agentic.turn":
                found_turn = True
            else:
                found_invocation = True

    assert found_turn, "Agentic turn span not found"
    assert found_invocation, "Agentic invocation span not found"


def verify_spans(memory_exporter, session_id: str):
    time.sleep(2)
    found_inference = found_agentic_turn = found_agent = found_tool = False
    spans = memory_exporter.get_finished_spans()
    for span in spans:
        span_attributes = span.attributes

        if span_attributes.get("span.type") in ("inference", "inference.framework"):
            # Inference spans come from the litellm metamodel, which OpenHands calls into
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == MODEL
            assert span_attributes["entity.2.type"] == f"model.llm.{MODEL}"

            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            found_inference = True

        if span_attributes.get("span.type") == "agentic.turn":
            assert span_attributes["entity.1.type"] == "agent.openhands"
            assert "entity.1.name" in span_attributes
            verify_input_output(span)
            found_agentic_turn = True

        if span_attributes.get("span.type") == "agentic.invocation":
            assert span_attributes["entity.1.type"] == "agent.openhands"
            assert "entity.1.name" in span_attributes
            verify_input_output(span)
            found_agent = True

        if span_attributes.get("span.type") == "agentic.tool.invocation":
            assert span_attributes["entity.1.type"] == "tool.openhands"
            # the tool span records the agent that decided to call the tool
            assert span_attributes["entity.2.type"] == "agent.openhands"
            verify_input_output(span)
            if span_attributes["entity.1.name"] == TerminalTool.name:
                found_tool = True

        if 'monocle_apptrace.version' in span_attributes:
            assert "scope.agentic.session" in span_attributes, f"scope.agentic.session not found in span {span.name}"
            assert span_attributes["scope.agentic.session"] == session_id, \
                f"Expected session {session_id}, got {span_attributes.get('scope.agentic.session')}"

    assert found_inference, "Inference span not found"
    assert found_agentic_turn, "Agentic turn span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, f"{TerminalTool.name} tool span not found"


def verify_input_output(span):
    span_input, span_output = span.events
    assert "input" in span_input.attributes
    assert span_input.attributes["input"] is not None and span_input.attributes["input"] != ""
    assert "response" in span_output.attributes
    assert span_output.attributes["response"] is not None and span_output.attributes["response"] != ""


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
