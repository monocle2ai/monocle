import logging
import os
import tempfile
import time

import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace import setup_monocle_telemetry
from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool
from openhands.sdk.context import Skill
from openhands.sdk.subagent import register_agent
from openhands.tools import register_builtins_agents
from openhands.tools.task import TaskToolSet
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


def create_math_helper(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        tools=[],
        agent_context=AgentContext(
            skills=[Skill(name="math", content="You answer arithmetic questions concisely.", trigger=None)],
            system_message_suffix="Only answer the arithmetic question.",
        ),
    )


def create_word_helper(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        tools=[],
        agent_context=AgentContext(
            skills=[Skill(name="words", content="You count letters in words precisely.", trigger=None)],
            system_message_suffix="Only answer the letter-counting question.",
        ),
    )


register_agent(name="math_helper", factory_func=create_math_helper, description="Answers arithmetic questions.")
register_agent(name="word_helper", factory_func=create_word_helper, description="Counts letters in words.")
register_builtins_agents()


def build_conversation(workspace_dir: str) -> Conversation:
    llm = LLM(
        model=MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        usage_id="openhands_test_llm",
    )
    # the task toolset is how OpenHands delegates work to a sub-agent; a concurrency
    # limit above 1 makes the delegated tools run on worker threads
    supervisor = Agent(llm=llm, tools=[Tool(name=TaskToolSet.name)], tool_concurrency_limit=2)
    return Conversation(agent=supervisor, workspace=workspace_dir)


def test_multi_agent(setup, workspace):
    conversation = build_conversation(workspace)
    conversation.send_message(
        "Delegate two tasks in parallel: ask the math_helper what 17 * 23 is, and ask the "
        "word_helper how many letters are in 'observability'. Then combine both answers "
        "in one sentence and finish."
    )
    conversation.run()
    verify_spans(setup, str(conversation.state.id))


def verify_spans(memory_exporter, session_id: str):
    time.sleep(2)
    found_inference = found_agent = found_task_tool = False
    found_supervisor_turn = found_delegated_turn = False
    spans = memory_exporter.get_finished_spans()
    spans_by_id = {span.context.span_id: span for span in spans}
    trace_ids = set()

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
            verify_input_output(span)
            parent_span = spans_by_id.get(span.parent.span_id) if span.parent else None
            parent_type = parent_span.attributes.get("span.type") if parent_span else None
            if parent_type == "agentic.tool.invocation":
                # a delegated sub-conversation runs nested under the task tool that
                # spawned it, not as a trace of its own
                found_delegated_turn = True
            else:
                found_supervisor_turn = True

        if span_attributes.get("span.type") == "agentic.invocation":
            assert span_attributes["entity.1.type"] == "agent.openhands"
            verify_input_output(span)
            found_agent = True

        if span_attributes.get("span.type") == "agentic.tool.invocation":
            assert span_attributes["entity.1.type"] == "tool.openhands"
            # the tool span records the agent that decided to call the tool
            assert span_attributes["entity.2.type"] == "agent.openhands"
            verify_input_output(span)
            if span_attributes["entity.1.name"] == "task":
                found_task_tool = True

        if 'monocle_apptrace.version' in span_attributes:
            trace_ids.add(span.context.trace_id)
            assert "scope.agentic.session" in span_attributes, f"scope.agentic.session not found in span {span.name}"
            # delegated sub-conversations inherit the parent session instead of
            # starting their own
            assert span_attributes["scope.agentic.session"] == session_id, \
                f"Expected session {session_id}, got {span_attributes.get('scope.agentic.session')}"

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_task_tool, "Task tool span not found"
    assert found_supervisor_turn, "Supervisor agentic turn span not found"
    assert found_delegated_turn, "Delegated agentic turn span not found"
    assert len(trace_ids) == 1, f"Expected the delegated work in one trace, got {len(trace_ids)}"


def verify_input_output(span):
    span_input, span_output = span.events
    assert "input" in span_input.attributes
    assert span_input.attributes["input"] is not None and span_input.attributes["input"] != ""
    assert "response" in span_output.attributes
    assert span_output.attributes["response"] is not None and span_output.attributes["response"] != ""


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
