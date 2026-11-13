# Evals in Monocle

An eval is an LLM-judge question asked of a trace: *did the agent report only what its tools
returned? did the supervisor follow its own routing rules?* The question lives in a template
file; the judge answers with one label from the template's list; the test asserts the label,
next to the span assertions it already makes.

With `with_evaluation("monocle")` the whole thing runs inside Monocle. The spans already in
the asserter are preprocessed into judge evidence and sent to the model named in the template.
You need one key: the model's.

```python
monocle_trace_asserter.called_tool("okahu_demo_cc_tool_process_refund").contains_output("submitted")

monocle_trace_asserter.with_evaluation("monocle", repeat=5) \
    .check_eval("templates/tool_output_utilization.json", expected="embellished")
```

## What is here

```
test_evals_in_monocle.py   three tests: two questions on a chat model, one of them on a decision model too
templates/                 the three templates those tests use
```

The trace is the customer-care refund run the test suite already ships
(`tests/integration/json_loader_traces/642dbd9d…json`): a supervisor reads
"Process a refund of $350 for order ORD-STD-0350" and hands it to three specialists in turn,
an order-lookup agent, an eligibility agent and a refund agent, each with one tool. The refund
tool returned only `{"submitted": true}`; the agent told the customer the refund was processed,
with a refund ID and a 3-5 day timeline.

| Template | Asks | Judge sees | Labels | On this trace |
|---|---|---|---|---|
| `tool_output_utilization` | Did the agent report its tools' results as returned, add to them, or get them wrong? | turn, agent and tool spans | faithful / embellished / misread | **embellished**, 5 of 5 runs on `openai:gpt-4.1-mini`: a refund ID, "processed" and a timeline no tool returned |
| `routing_policy_adherence` | Did the supervisor follow the routing rules in its own system prompt? | the default scope (no `evidence_scope` in the file): every agentic and inference span, all roles, so the system message is in view | adhered / deviated | **adhered**: lookup, then eligibility, then refund, as the rules say |
| `tool_output_utilization_decision` | the first question on a decision model | as the first row | the same, plus a probability per label | **embellished** at probability 0.98 on `typesafe:jev-latest` |

The two questions look at the same run from different sides: the routing was right, and the
answer still said more than the tools knew. Each test also makes the span assertions that pin
what happened before the judge is asked what to make of it: `called_tool` with the agent that
called it, `contains_output` on a tool's result, `called_agent(..., count=1)`,
`does_not_call_agent`, `under_token_limit`, `under_duration`.

One verdict per trace. When the asserter holds several traces, `check_eval` compares every
one of them to `expected`; `expected` can be a list when more than one label is acceptable.

## The template

The custom template shape `check_eval` already accepts, plus two blocks:

```json
"evidence_scope": {
    "span_types": ["agentic.turn", "agentic.invocation", "agentic.tool.invocation"],
    "roles": ["user", "assistant", "tool"],
    "events": ["input", "output"]
},
"llm": {
    "model": "openai:gpt-4.1-mini",
    "params": {"temperature": 0, "max_tokens": 2000}
}
```

`evidence_scope` decides which spans, message roles and events the judge sees; narrow it to
what the question needs. `tool_output_utilization` leaves inference spans out because the tool
outputs are all it needs. `routing_policy_adherence` has no `evidence_scope` at all and gets the
default: every agentic and inference span type, all four roles, input and output, which is
what puts the system message, and the rules in it, in front of the judge. Inference-only scopes
get a short preamble about messages; any scope with agent or tool spans gets the preamble
about the agent's execution tree.

`llm.model` is a `provider:model` string for LangChain's `init_chat_model`, or
`<provider>:<model>` for a decision model with `min_confidence` (`typesafe:` today; a provider is one
class in `judge.py`'s `DECISION_JUDGES`). The template is the one place
the model and the scope are set; the assertion does not override them.

The same file runs unchanged with `with_evaluation("okahu")`; the Monocle-only `llm` keys are
dropped on the way.

The rules the file must follow, the inconclusive-result messages, and the full `llm` block are
in [docs/monocle_evaluation_api.md](../../../docs/monocle_evaluation_api.md#12-running-evals-in-monocle).

## Judges, repeats, confidence

A chat model is not deterministic. `with_evaluation("monocle", repeat=5)` runs the judge five
times and asserts the majority label; when fewer than 80% of the runs agree the test fails as
inconclusive, with its own message, so a flaky verdict is not mistaken for a wrong one.

Labels have to be mutually exclusive, or two judges will split them differently. The first
draft of `tool_output_utilization` ranked its labels ("when both apply, misread wins"); the chat
model answered `embellished` and the decision model `misread`, each with a reason. Defining the
two as different kinds of mistake, rather than ranking them, brought both judges to the same
answer. A verdict that moves between judges or runs is a prompt to sharpen, not a label to pin.

A decision model answers once, with a probability per label. `min_confidence` in
the template fails the test when the chosen label's probability is under it. `repeat` is
rejected for a decision model: the same input gives the same answer.

## Run

```bash
cd test_tools
export OPENAI_API_KEY=...       # the chat-model templates name openai:gpt-4.1-mini
export TYPESAFE_API_KEY=...     # the decision-model template names typesafe:jev-latest
pytest examples/evals -v
```

Each test skips unless the key for its template's provider is set: the part of `llm.model`
before the colon, upper-cased, plus `_API_KEY`. (`TYPESAFE_API_URL` overrides that provider's
endpoint when it is not `https://api.typesafe.ai`.)

The labels in the table are what `openai:gpt-4.1-mini` and `jev-1.14.0` returned on the
first live run (4 Nov 2026). If a judge disagrees on your run, read its `explanation` in the
failure message before changing either the template or the expectation.

## Writing your own

1. Record a trace, or point `with_trace_source` at one.
2. Copy the template closest to your question. Keep `label` and `explanation`; add output
   fields only when a test will read them. Make the labels mutually exclusive.
3. Set `evidence_scope` to the smallest set of spans that can answer the question.
4. Run it once, read the explanation, then pin `expected`.
