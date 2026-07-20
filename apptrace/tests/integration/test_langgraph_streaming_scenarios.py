"""LangGraph **streaming** agent scenarios.

Each test function is a self-contained, realistic streaming LangGraph agent
behaviour that produces a *distinct* trace in Okahu.  They deliberately avoid
duplicating the basic ``values`` / ``updates`` / ``messages`` stream-mode tests
already covered in ``test_langgraph_stream_sample.py`` — every scenario below
adds an agentic behaviour that changes the *shape* of the emitted trace
(sequential vs parallel tool loops, an LLM nested inside a tool, RAG retrieval,
conversation memory, supervisor delegation, peer handoff, collaboration).

Scenarios in this file:
    1. test_stream_sequential_dependent_tools . agent calls tool A then tool B (details -> price), streamed
    2. test_stream_parallel_tool_calls ........ one agent step requests several tools at once, streamed
    3. test_stream_nested_tool_llm ............ a tool whose body calls ChatOpenAI -> inference span nested under a tool
    4. test_stream_rag ........................ a retriever tool over a tiny in-memory corpus -> grounded streamed answer
    5. test_stream_memory_multi_turn .......... MemorySaver checkpointer + thread_id; turn 2 recalls turn 1 (>=2 inference spans)
    6. test_stream_supervisor_delegation ...... supervisor delegates to worker agents (langgraph-supervisor), streamed
    7. test_stream_peer_handoff ............... agent A hands control peer-to-peer to agent B (handoff tool), streamed
    8. test_stream_multi_agent_collaboration .. researcher + writer agents collaborate in one streamed supervisor run

All scenarios reuse ``common.stream_helpers.build_stream_span_processors`` for
Okahu + in-memory wiring and ``common.helpers`` for span verification.  Streams
are drained fully, then we sleep ~2s to let the SimpleSpanProcessor flush before
asserting on the captured spans.

NOTE on APIs (verified against the installed versions in this env):
  * langchain 1.2.15          -> ``from langchain.agents import create_agent``
  * langgraph 1.1.5           -> StateGraph / MessagesState / START / END
  * langgraph-prebuilt 1.0.13 -> ``create_react_agent`` (used for supervisor workers)
  * langgraph-checkpoint 4.1  -> ``MemorySaver``
  * langgraph-supervisor 0.0.31 -> ``create_supervisor`` (delegation/collaboration)
"""

import asyncio
import logging
import time

import pytest
from common.helpers import find_spans_by_type, verify_inference_span
from common.stream_helpers import build_stream_span_processors
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"


@pytest.fixture(scope="function")
def setup():
    exporter, span_processors = build_stream_span_processors()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="langgraph_streaming_scenarios",
            span_processors=span_processors,
        )
        yield exporter
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def _llm():
    return ChatOpenAI(model=MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Local tool + corpus definitions (kept tiny; the point is the trace shape).
# ---------------------------------------------------------------------------

COFFEE_MENU = {
    "espresso": "A strong and bold coffee shot.",
    "latte": "A smooth coffee with steamed milk.",
    "cappuccino": "A rich coffee with frothy milk foam.",
    "americano": "Espresso with added hot water for a milder taste.",
    "mocha": "A chocolate-flavored coffee with whipped cream.",
}

COFFEE_PRICES = {
    "espresso": 3.0,
    "latte": 4.5,
    "cappuccino": 4.0,
    "americano": 3.5,
    "mocha": 5.0,
}

# Tiny in-memory "vector DB" for the RAG scenario: id -> document text.
KNOWLEDGE_BASE = {
    "doc1": "Okahu is an observability platform for GenAI applications and agents.",
    "doc2": "Monocle is an open-source project that instruments GenAI apps to emit traces.",
    "doc3": "LangGraph is a library for building stateful, multi-actor agent applications.",
    "doc4": "Espresso is brewed by forcing hot water through finely-ground coffee beans.",
}


@tool
def get_coffee_details(coffee_name: str) -> str:
    """Provides details about a specific coffee. Input the coffee name."""
    return COFFEE_MENU.get(coffee_name.lower(), "Sorry, we don't have details for that coffee.")


@tool
def get_coffee_price(coffee_name: str) -> str:
    """Provides the price in USD of a specific coffee. Input the coffee name."""
    price = COFFEE_PRICES.get(coffee_name.lower())
    if price is None:
        return "Sorry, we don't have a price for that coffee."
    return f"${price:.2f}"


@tool
def summarize_coffee(coffee_name: str) -> str:
    """Writes a short marketing tagline for a coffee. Uses an LLM internally."""
    # Nested LLM call inside a tool -> the trace shows an inference span
    # nested under this tool's invocation span.
    details = COFFEE_MENU.get(coffee_name.lower(), coffee_name)
    resp = _llm().invoke(
        [HumanMessage(content=f"Write a 6-word catchy tagline for: {details}")]
    )
    return resp.content


@tool
def retrieve_documents(query: str) -> str:
    """Retrieves relevant documents from the knowledge base for a query."""
    # Naive keyword "retriever" over the in-memory corpus (no external vector DB).
    q_words = {w.lower().strip(".,?!") for w in query.split()}
    hits = []
    for doc_id, text in KNOWLEDGE_BASE.items():
        text_words = {w.lower().strip(".,?!") for w in text.split()}
        if q_words & text_words:
            hits.append(f"[{doc_id}] {text}")
    if not hits:
        # Fall back to returning the whole (tiny) corpus so the agent has context.
        hits = [f"[{doc_id}] {text}" for doc_id, text in KNOWLEDGE_BASE.items()]
    return "\n".join(hits)


# ---------------------------------------------------------------------------
# Local verify helpers (mirror verify_stream_spans in test_langgraph_stream_sample).
# ---------------------------------------------------------------------------
def _assert_inference_spans(exporter, min_count=1):
    """Assert >= min_count LangGraph/OpenAI inference spans were captured."""
    spans = exporter.get_captured_spans()
    inference_spans = find_spans_by_type(spans, "inference") + find_spans_by_type(
        spans, "inference.framework"
    )
    assert len(inference_spans) >= min_count, (
        f"Expected >= {min_count} inference spans, got {len(inference_spans)}. "
        f"Span types seen: {sorted({s.attributes.get('span.type') for s in spans})}"
    )
    for span in inference_spans:
        verify_inference_span(
            span=span,
            entity_type="inference.openai",
            model_name=MODEL,
            model_type=f"model.llm.{MODEL}",
            check_metadata=True,
            check_input_output=True,
        )
    return inference_spans


def _assert_agent_spans(exporter):
    """Assert LangGraph agent invocation/turn spans were captured (agent.langgraph)."""
    spans = exporter.get_captured_spans()
    agent_spans = find_spans_by_type(spans, "agentic.invocation")
    turn_spans = find_spans_by_type(spans, "agentic.turn")
    assert agent_spans or turn_spans, (
        "Expected agentic.invocation or agentic.turn spans. "
        f"Span types seen: {sorted({s.attributes.get('span.type') for s in spans})}"
    )
    for span in agent_spans:
        if "entity.1.name" in span.attributes:
            assert span.attributes.get("entity.1.type") == "agent.langgraph"
    return agent_spans, turn_spans


def _assert_tool_spans(exporter, min_count=1):
    """Assert >= min_count LangGraph tool invocation spans (tool.langgraph)."""
    spans = exporter.get_captured_spans()
    tool_spans = find_spans_by_type(spans, "agentic.tool.invocation")
    assert len(tool_spans) >= min_count, (
        f"Expected >= {min_count} tool spans, got {len(tool_spans)}. "
        f"Span types seen: {sorted({s.attributes.get('span.type') for s in spans})}"
    )
    for span in tool_spans:
        assert span.attributes.get("entity.1.type") == "tool.langgraph"
        assert "entity.1.name" in span.attributes
    return tool_spans


def _drain_sync(agent, content, stream_mode="values", config=None):
    """Drain a synchronous agent stream, returning the chunk count."""
    n = 0
    for chunk in agent.stream(
        input={"messages": [HumanMessage(content=content)]},
        stream_mode=stream_mode,
        config=config,
    ):
        n += 1
        logger.info("sync chunk %d: %s", n, type(chunk).__name__)
    return n


async def _drain_async(agent, content, stream_mode="values", config=None):
    """Drain an async agent stream, returning the chunk count."""
    n = 0
    async for chunk in agent.astream(
        input={"messages": [HumanMessage(content=content)]},
        stream_mode=stream_mode,
        config=config,
    ):
        n += 1
        logger.info("async chunk %d: %s", n, type(chunk).__name__)
    return n


# ---------------------------------------------------------------------------
# 1. Sequential dependent tool calls — details (A) then price (B).
# ---------------------------------------------------------------------------
def test_stream_sequential_dependent_tools(setup):
    """Unique: two *dependent* tool rounds (get_coffee_details -> get_coffee_price)
    inside one streamed agent run, producing two distinct tool spans in sequence."""
    agent = create_agent(_llm(), [get_coffee_details, get_coffee_price])
    n = _drain_sync(
        agent,
        "First get the details of a latte, then tell me its price. "
        "Use one tool at a time.",
        stream_mode="values",
    )
    assert n > 0
    time.sleep(2)
    _assert_inference_spans(setup, min_count=2)
    _assert_agent_spans(setup)
    _assert_tool_spans(setup, min_count=1)


# ---------------------------------------------------------------------------
# 2. Parallel tool calls — one agent step emits several tool calls at once.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio(loop_scope="function")
async def test_stream_parallel_tool_calls(setup):
    """Unique: a single streamed agent step that fans out into multiple tool
    calls (details AND price for the same drink) executed in one turn."""
    agent = create_agent(_llm(), [get_coffee_details, get_coffee_price])
    n = await _drain_async(
        agent,
        "In one step, get BOTH the details and the price for an espresso.",
        stream_mode="values",
    )
    assert n > 0
    await asyncio.sleep(2)
    _assert_inference_spans(setup, min_count=1)
    _assert_agent_spans(setup)
    _assert_tool_spans(setup, min_count=1)


# ---------------------------------------------------------------------------
# 3. Nested tool call — a tool whose body itself invokes an LLM.
# ---------------------------------------------------------------------------
def test_stream_nested_tool_llm(setup):
    """Unique: the ``summarize_coffee`` tool calls ChatOpenAI internally, so the
    trace shows an inference span *nested under* a tool invocation span (in
    addition to the agent's own reasoning inference spans)."""
    agent = create_agent(_llm(), [summarize_coffee])
    n = _drain_sync(
        agent,
        "Use the summarize tool to create a tagline for a mocha.",
        stream_mode="values",
    )
    assert n > 0
    time.sleep(2)
    # >=2 inference spans: the agent's reasoning turn(s) + the LLM call inside the tool.
    _assert_inference_spans(setup, min_count=2)
    _assert_agent_spans(setup)
    _assert_tool_spans(setup, min_count=1)


# ---------------------------------------------------------------------------
# 4. RAG streaming — retriever tool over a tiny in-memory corpus.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio(loop_scope="function")
async def test_stream_rag(setup):
    """Unique: a retrieval-augmented run — the ``retrieve_documents`` tool pulls
    grounding docs from an in-memory corpus, then the agent streams an answer
    grounded in them (retrieval tool span + grounded inference spans)."""
    agent = create_agent(
        _llm(),
        [retrieve_documents],
        system_prompt=(
            "You answer questions using ONLY the documents returned by the "
            "retrieve_documents tool. Always call the tool first."
        ),
    )
    n = await _drain_async(
        agent,
        "What is Monocle and what does it do?",
        stream_mode="values",
    )
    assert n > 0
    await asyncio.sleep(2)
    _assert_inference_spans(setup, min_count=1)
    _assert_agent_spans(setup)
    _assert_tool_spans(setup, min_count=1)


# ---------------------------------------------------------------------------
# 5. Memory retrieval / conversation history — MemorySaver + thread_id.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio(loop_scope="function")
async def test_stream_memory_multi_turn(setup):
    """Unique: two streamed turns sharing a MemorySaver checkpointer + thread_id,
    so turn 2 recalls the coffee named in turn 1 without it being repeated.
    Multiple turns -> multiple inference spans in the same conversation thread."""
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    agent = create_agent(
        _llm(),
        [get_coffee_details],
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "coffee-thread-1"}}

    # Turn 1: establish context.
    n1 = await _drain_async(
        agent,
        "My favorite coffee is a cappuccino. Remember that.",
        stream_mode="values",
        config=config,
    )
    # Turn 2: rely on memory of turn 1 (note: the drink is NOT named here).
    n2 = await _drain_async(
        agent,
        "What are the details of my favorite coffee?",
        stream_mode="values",
        config=config,
    )
    assert n1 > 0 and n2 > 0
    await asyncio.sleep(2)
    # Two conversational turns -> at least two inference spans.
    _assert_inference_spans(setup, min_count=2)
    _assert_agent_spans(setup)


# ---------------------------------------------------------------------------
# 6. Supervisor delegation — supervisor delegates to specialized worker agents.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio(loop_scope="function")
async def test_stream_supervisor_delegation(setup):
    """Unique: a supervisor agent delegates sub-tasks to two worker agents
    (a menu expert and a pricing expert) via langgraph-supervisor handoff tools,
    then control returns to the supervisor. Streamed. The trace shows the
    supervisor -> worker -> back delegation topology (multiple agent spans)."""
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    menu_agent = create_react_agent(
        _llm(),
        tools=[get_coffee_details],
        name="menu_expert",
        prompt="You are a coffee menu expert. Use get_coffee_details to answer.",
    )
    pricing_agent = create_react_agent(
        _llm(),
        tools=[get_coffee_price],
        name="pricing_expert",
        prompt="You are a coffee pricing expert. Use get_coffee_price to answer.",
    )
    supervisor = create_supervisor(
        agents=[menu_agent, pricing_agent],
        model=_llm(),
        prompt=(
            "You are a supervisor coordinating a menu_expert and a pricing_expert. "
            "Delegate questions about coffee details to menu_expert and questions "
            "about coffee prices to pricing_expert."
        ),
    ).compile()

    n = await _drain_async(
        supervisor,
        "Tell me the details of a latte and also its price.",
        stream_mode="values",
    )
    assert n > 0
    await asyncio.sleep(2)
    _assert_inference_spans(setup, min_count=2)
    _assert_agent_spans(setup)
    _assert_tool_spans(setup, min_count=1)


# ---------------------------------------------------------------------------
# 7. Agent handoff — peer-to-peer transfer of control (A -> B), no return.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio(loop_scope="function")
async def test_stream_peer_handoff(setup):
    """Unique: a *peer-to-peer* handoff built on a bare StateGraph. A greeter
    agent handles the opening, then hands control directly to a barista agent
    which produces the final answer — control does NOT bounce back to a
    supervisor (contrast with test_stream_supervisor_delegation). Streamed."""
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import create_react_agent

    greeter = create_react_agent(
        _llm(),
        tools=[],
        name="greeter",
        prompt="You are a friendly greeter. Greet the user in one short sentence.",
    )
    barista = create_react_agent(
        _llm(),
        tools=[get_coffee_details],
        name="barista",
        prompt="You are a barista. Use get_coffee_details to describe the coffee asked about.",
    )

    async def greeter_node(state: MessagesState):
        result = await greeter.ainvoke(state)
        return {"messages": result["messages"]}

    async def barista_node(state: MessagesState):
        result = await barista.ainvoke(state)
        return {"messages": result["messages"]}

    # Static peer handoff: greeter -> barista -> END (A always hands off to B).
    graph = StateGraph(MessagesState)
    graph.add_node("greeter", greeter_node)
    graph.add_node("barista", barista_node)
    graph.add_edge(START, "greeter")
    graph.add_edge("greeter", "barista")
    graph.add_edge("barista", END)
    app = graph.compile()

    n = await _drain_async(
        app,
        "Hi! Can you tell me about a cappuccino?",
        stream_mode="values",
    )
    assert n > 0
    await asyncio.sleep(2)
    # Two agents each run an inference turn -> >=2 inference spans.
    _assert_inference_spans(setup, min_count=2)
    _assert_agent_spans(setup)


# ---------------------------------------------------------------------------
# 8. Multi-agent collaboration — researcher + writer collaborate in one run.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio(loop_scope="function")
async def test_stream_multi_agent_collaboration(setup):
    """Unique: two agents *collaborate* on one task in a single streamed run — a
    researcher gathers grounding facts via the retriever, then a writer composes
    the final prose from those facts. Coordinated by a supervisor so both agents
    contribute to the same output (contrast with delegation, where each worker
    answers an independent sub-question)."""
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    researcher = create_react_agent(
        _llm(),
        tools=[retrieve_documents],
        name="researcher",
        prompt=(
            "You are a researcher. Use retrieve_documents to gather relevant facts "
            "and report them plainly. Do not write prose."
        ),
    )
    writer = create_react_agent(
        _llm(),
        tools=[],
        name="writer",
        prompt=(
            "You are a writer. Turn the facts provided by the researcher into a "
            "single polished paragraph."
        ),
    )
    supervisor = create_supervisor(
        agents=[researcher, writer],
        model=_llm(),
        prompt=(
            "You coordinate a researcher and a writer. First have the researcher "
            "gather facts, then have the writer turn those facts into a paragraph. "
            "Return the writer's paragraph as the final answer."
        ),
    ).compile()

    n = await _drain_async(
        supervisor,
        "Write a short paragraph explaining what Monocle and LangGraph are.",
        stream_mode="values",
    )
    assert n > 0
    await asyncio.sleep(2)
    # Supervisor + researcher + writer each run inference -> several inference spans.
    _assert_inference_spans(setup, min_count=2)
    _assert_agent_spans(setup)
    _assert_tool_spans(setup, min_count=1)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
