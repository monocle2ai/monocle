"""One recorded trace, checked with span assertions and judged inside Monocle with only a model key.

The trace is a LangGraph customer-care run. A supervisor reads "Process a refund of $350 for
order ORD-STD-0350" and hands it to three specialists in turn: an order-lookup agent, an
eligibility agent and a refund agent, each with one tool. The refund tool returned only
{"submitted": true}; the agent told the customer the refund was processed, with a refund ID
and a 3-5 day timeline.

Each test skips unless the key for its template's llm.model is set: the provider before the
colon, upper-cased, plus _API_KEY. "openai:gpt-4.1-mini" needs OPENAI_API_KEY. The last test
needs no key: model="local" judges on an open model on this machine.

Run from test_tools:
    pytest examples/evals -v
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest
from monocle_test_tools import TraceAssertion
from monocle_test_tools.file_span_loader import JSONSpanLoader
from monocle_test_tools.harness.local_model import install_command

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"
TRACE = HERE.parents[1] / "tests" / "integration" / "json_loader_traces" / "642dbd9d0dfcfdbdc8849f67f34c8a19.json"

SUPERVISOR = "okahu_demo_cc_agent_supervisor"
ORDER_LOOKUP = "okahu_demo_cc_agent_order_lookup"
ELIGIBILITY = "okahu_demo_cc_agent_eligibility"
REFUND = "okahu_demo_cc_agent_refund"


def refund_trace_spans() -> list:
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    return JSONSpanLoader.from_json_str(json.dumps(trace["spans"]))


def template(name: str) -> str:
    return str(TEMPLATES / f"{name}.json")


def needs_model_key(name: str):
    model = json.loads(Path(template(name)).read_text(encoding="utf-8"))["llm"]["model"]
    key = f"{model.partition(':')[0].upper()}_API_KEY"
    return pytest.mark.skipif(not os.getenv(key), reason=f"needs {key} for {model}")


needs_local_runtime = pytest.mark.skipif(
    importlib.util.find_spec("llama_cpp") is None, reason=f"needs llama-cpp-python: {install_command()}"
)


@needs_model_key("tool_output_utilization")
def test_agent_reports_only_what_its_tools_returned(monocle_trace_asserter: TraceAssertion):
    """The span assertions pin what happened: which tools ran, under which agent, what came back.
    The eval then asks what the agent made of those results. Here the tools did their job and the
    agent still added a refund ID and a timeline no tool returned: 'embellished'.

    repeat=5 runs the judge five times and asserts the majority label; fewer than 80% agreement
    fails the test as inconclusive rather than as a wrong label."""
    monocle_trace_asserter.load_spans(refund_trace_spans())

    monocle_trace_asserter.called_tool("okahu_demo_cc_tool_lookup_order", ORDER_LOOKUP) \
        .contains_input("ORD-STD-0350") \
        .contains_output("350")
    monocle_trace_asserter.called_tool("okahu_demo_cc_tool_process_refund", REFUND) \
        .contains_output("submitted") \
        .does_not_contain_output("REF-")
    monocle_trace_asserter.contains_output("REF-")

    monocle_trace_asserter.with_evaluation("monocle", repeat=5) \
        .check_eval(template("tool_output_utilization"), expected="embellished")


@needs_model_key("routing_policy_adherence")
def test_supervisor_follows_its_own_routing_rules(monocle_trace_asserter: TraceAssertion):
    """The supervisor's system prompt is a routing rulebook. The span assertions check the shape
    of the run: three hand-offs, each specialist once, nothing else called, within budget. The
    eval reads the rulebook from the trace and judges the hand-offs against it: 'adhered'.

    This template has no evidence_scope, so the judge gets the default scope: every agentic and
    inference span type, all four message roles, input and output. That is what puts the system
    message in view. The first template narrows its scope because the tool outputs are all it needs."""
    monocle_trace_asserter.load_spans(refund_trace_spans())

    monocle_trace_asserter.called_agent(SUPERVISOR)
    monocle_trace_asserter.called_agent(ORDER_LOOKUP, count=1)
    monocle_trace_asserter.called_agent(ELIGIBILITY, count=1)
    monocle_trace_asserter.called_agent(REFUND, count=1)
    monocle_trace_asserter.does_not_call_agent("okahu_demo_cc_agent_shipping")
    monocle_trace_asserter.does_not_call_tool("okahu_demo_cc_tool_get_return_policy")
    monocle_trace_asserter.under_token_limit(10_000)
    monocle_trace_asserter.under_duration(15, units="seconds")

    monocle_trace_asserter.with_evaluation("monocle") \
        .check_eval(template("routing_policy_adherence"), expected="adhered")


@needs_model_key("tool_output_utilization_decision")
def test_tool_output_utilization_on_a_decision_model(monocle_trace_asserter: TraceAssertion):
    """The first question again, on a decision model: the same prompt and scope, with the
    template's llm.model naming the decision provider. A decision model answers once with a
    probability per label and no prose; min_confidence marks the verdict inconclusive when the
    chosen label's probability is under it. repeat is not allowed here: the same input gives the
    same answer."""
    monocle_trace_asserter.load_spans(refund_trace_spans())

    monocle_trace_asserter.with_evaluation("monocle") \
        .check_eval(template("tool_output_utilization_decision"), expected="embellished")


@needs_local_runtime
def test_agent_reports_only_what_its_tools_returned_on_a_local_model(monocle_trace_asserter: TraceAssertion):
    """The first question again, with no API key: model="local" runs the judge on Qwen3-4B on this
    machine through llama.cpp, in place of the model the template names. Everything else is the
    same template and the same evidence. The first run downloads the model (about 2.5 GB) and
    starts the server; each verdict takes a few seconds on an Apple-silicon GPU."""
    monocle_trace_asserter.load_spans(refund_trace_spans())

    monocle_trace_asserter.with_evaluation("monocle", repeat=3, model="local") \
        .check_eval(template("tool_output_utilization"), expected="embellished")
