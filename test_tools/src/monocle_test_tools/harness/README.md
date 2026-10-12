# Scenario Test Harness

The scenario harness tests an agent by **talking to it**. Instead of scripting every
turn, you describe a scenario — what you're testing, the user persona to play, the
details of the request — and an LLM-backed *test agent* holds the conversation with your
*target agent* while an LLM judge decides whether the scenario succeeded.

It answers questions a scripted test cannot:

- Does the agent *ask* for information it's missing, or invent it?
- Does it stay on task with an impatient, terse, or rude user?
- Does it actually complete the job, or just promise to?

This is the dynamic counterpart to `MultiTurnTestCase`. Use `MultiTurnTestCase` when you
know every turn in advance; use the harness when the point is that you *don't*.

---

## Installation

The harness needs LangChain, which is an optional extra:

```bash
pip install "monocle_test_tools[test_agent]"
```

Importing `monocle_test_tools` without it still works — the harness imports LangChain
lazily, so only the harness itself requires the extra.

---

## Quick start

```python
import pytest
from monocle_test_tools.harness import ScenarioHarness, ScenarioTestCase
from monocle_test_tools.runner.runner import AgentTypes

CASE = {
    "test_name": "book_flight_foul_mouthed_user",
    "scenario": "Book a flight from source to destination.",
    "persona": "A foul mouthed, impatient user who gives one detail at a time",
    "params": [
        {"name": "source",      "value": "San Francisco",     "is_initial": True,
         "description": "source airport"},
        {"name": "destination", "value": "Seattle",           "is_initial": False,
         "description": "destination airport"},
        {"name": "date",        "value": "22nd October 2026", "is_initial": False,
         "description": "travel date"},
    ],
    "success_criteria": "The agent confirms that the flight has been booked.",
    "max_turns": 8,
}

@pytest.mark.asyncio
async def test_books_a_flight(monocle_trace_asserter):
    harness = ScenarioHarness(ScenarioTestCase.model_validate(CASE))

    result = await harness.run_scenario_async(
        my_agent, AgentTypes.LANGGRAPH,
        target_description="A travel booking agent that books flights between airports "
                           "for a given date, and hotels in a city for a given date.")

    assert result.passed, result.report()
    monocle_trace_asserter.called_tool("book_flight")   # trace assertions still apply
```

The scenario spec is plain JSON, so keeping cases in a file needs no special loader:

```python
with open("scenario_test_cases.json", encoding="utf-8") as handle:
    cases = json.load(handle)

@pytest.mark.asyncio
@pytest.mark.parametrize("case", cases, ids=lambda c: c["test_name"])
async def test_scenario(case, monocle_trace_asserter):
    harness = ScenarioHarness(ScenarioTestCase.model_validate(case))
    ...
```

A complete worked example lives in
[`test_tools/tests/integration/test_scenario_harness.py`](../../../tests/integration/test_scenario_harness.py),
with its target agent in
[`test_tools/tests/test_common/langgraph_travel_agent.py`](../../../tests/test_common/langgraph_travel_agent.py).

---

## How a scenario runs

The **driver** owns the loop. Each turn it asks the test agent for the next user message,
sends that to your target agent through `MonocleValidator.run_agent_async`, and judges
the reply:

```
for turn in 1..max_turns:
    tester_msg    = test agent writes the next user message (in persona)
    target_reply  = run_agent_async(target, tester_msg, session_id=...)   # spans captured
    verdict       = judge(target_reply, success_criteria)
    if verdict.met: PASS
FAIL("max turns exhausted")
```

Three properties follow from that shape:

- **One session for the whole scenario.** A single `session_id` is threaded into every
  call, so your agent keeps its own memory across turns (LangGraph `thread_id`, ADK
  session, and so on), and each turn's spans are tagged with `scope.turn_id`.
- **The judge runs every turn and cannot be skipped.** Its verdict is the loop's exit
  condition and the test's result. The test agent also gets the same judge as an
  `evaluate_response` tool, but that copy is advisory — it only shapes what the tester
  says next.
- **The test agent emits no spans.** Its LLM calls, and the judge's, run under the
  `MONOCLE_SUPPRESS_SPANS` context flag, so `result.spans` holds target spans only and
  your trace assertions see nothing but the agent under test.

---

## The scenario spec

| Field | Type | Default | Meaning |
|---|---|---|---|
| `test_name` | `str` | `"monocle_scenario_test"` | Name of the scenario. |
| `scenario` | `str` | *required* | What is being tested. |
| `persona` | `str` | *required* | The user the test agent plays. |
| `params` | `list[ScenarioParam]` | `[]` | Details of the request (see below). |
| `success_criteria` | `str` | *required* | Free text, judged by an LLM. |
| `max_turns` | `int` | `10` | Maximum target-agent invocations. |
| `session_id` | `str` | auto | Shared by every turn; generated when omitted. |

Validation rejects an empty `scenario` or `success_criteria`, `max_turns < 1`, duplicate
param names, param names that aren't valid identifiers (they become tool names), and a
non-empty `params` list with no `is_initial` param — the test agent would have nothing to
open the conversation with.

### Parameters and progressive disclosure

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | *required* | Becomes the tool name; must be an identifier. |
| `value` | `Any` | *required* | What the tool returns. |
| `description` | `str` | *required* | Becomes the tool description. |
| `is_initial` | `bool` | `False` | Stated in the opening message. |
| `required` | `bool` | `True` | The target must ask for it (see below). |

Every param becomes a **zero-argument tool** on the test agent. The preamble tells it to
call a tool *only* when the target agent asks for that detail:

```
turn 1  tester> I need a damn flight out of San Francisco.
        target> Where would you like to fly to?
turn 2  tester> [called destination()] ==> Seattle
        tester> Seattle, obviously.
        target> What date?
turn 3  tester> [called date()] ==> 22nd October 2026
        tester> 22nd October 2026. Get on with it.
```

The important part: **the value of a non-initial param never enters the prompt** — only
its name and description do. The test agent therefore *cannot* state the destination
without calling the tool first. Progressive disclosure is structural, not a matter of the
model following instructions.

Word `description` the way the *target agent* would refer to the detail, so the test
agent can match the target's question to the right tool.

### `required` — catching invented details

A `required` param whose tool was never called fails the scenario **even when the judge
says the criteria were met**. That combination is the signature of an agent satisfying
the success criteria with a detail it made up instead of asking for:

```
FAILED scenario 'Book a flight from source to destination.' after 4/8 turns
  reason: required params never requested: date
  never requested: date
  last verdict: The agent explicitly confirms that the flight has been booked.
  ...
```

The judge is telling the truth — a flight *was* booked. It was booked for a date nobody
ever supplied.

`is_initial` params are exempt: their values go out in the opening message, so there is
nothing left for the target to ask for. Set `required: false` for a detail the target is
free to skip.

> A "never requested" failure is worth reading carefully, because it can indict either
> side. A target agent that invents a value is a real bug in your agent. A *test agent*
> that invents one is a bug in the harness — check the preamble and what the driver puts
> in the tester's history before blaming the target.

---

## API

### `ScenarioHarness(case, model=None, judge_model=None, trace_test_agent=None)`

- **`case`** — a `ScenarioTestCase` or a plain `dict` (validated for you).
- **`model`** — the test agent's LLM: a model string or a built `BaseChatModel`.
  Defaults to `MONOCLE_TEST_AGENT_MODEL`, resolved with `init_chat_model`.
- **`judge_model`** — the judge's LLM. Defaults to `MONOCLE_TEST_JUDGE_MODEL`, falling
  back to the test agent's model.
- **`trace_test_agent`** — set `True` to stop suppressing the test agent's spans when
  you are debugging the tester itself. Defaults to `MONOCLE_TRACE_TEST_AGENT`.

Passing models explicitly is the route to take when a provider needs constructor
arguments `init_chat_model` can't supply from the environment — Azure OpenAI, for
instance:

```python
harness = ScenarioHarness(case,
                          model=AzureChatOpenAI(..., temperature=0.7),
                          judge_model=AzureChatOpenAI(..., temperature=0))
```

### `await harness.run_scenario_async(target_agent, agent_type, target_description)`

Runs the scenario and returns a `ScenarioResult`. It does **not** raise on a behavioral
failure — a target-agent exception is caught, recorded in `failure_reason`, and ends the
loop.

- **`target_agent`** — whatever the runner for `agent_type` expects (an agent object, a
  URL, …).
- **`agent_type`** — any value `get_agent_runner` accepts: `AgentTypes.LANGGRAPH`,
  `GOOGLE_ADK`, `CREWAI`, `STRANDS`, `LLAMAINDEX`, `OPENAI`, `MSAGENT`, `HTTP`, …
- **`target_description`** — what the target agent does, in a sentence or two. This goes
  into the test agent's system prompt, so it's how the tester knows what it can
  reasonably ask for.

### `await harness.test_scenario_async(...)`

Same arguments; runs the scenario, then asserts `result.passed` with `result.report()` as
the message. Use `run_scenario_async` when you want to inspect the result (including
negative tests, where a failure is the expected outcome).

### `ScenarioResult`

| Field | Meaning |
|---|---|
| `passed` | Judge satisfied **and** no required param left unrequested. |
| `turns_used` / `max_turns` | Turns actually taken, and the budget. |
| `failure_reason` | `"max turns exhausted"`, `"target error: …"`, and/or the required-param clause. |
| `turns` | `TurnRecord` per turn: `tester_message`, `target_response`, `verdict`, `param_tools_called` (a `ParamCall` per tool call, with `name` and the `value` it returned). |
| `missing_required_params` | Required params the target never asked for. |
| `spans` | Target spans across the whole scenario. |
| `per_turn_spans` | Target spans, one tuple per turn. |
| `report()` | The whole run as a readable transcript — use it as your assert message. |

`turns`, `spans` and `per_turn_spans` are excluded from the model's `repr`. A failing
`assert result.passed` makes pytest print `repr(result)`, and dumping every span object
and message there would bury the failure under hundreds of lines. They remain fully
available as attributes — the transcript is what `report()` is for.

At the end of a run the harness points the validator's span pool at every target span the
scenario produced, so the `monocle_trace_asserter` fixture asserts across the whole
conversation rather than just the last turn.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `MONOCLE_TEST_AGENT_MODEL` | Test agent model, e.g. `openai:gpt-4.1`. Resolved by `init_chat_model`. |
| `MONOCLE_TEST_JUDGE_MODEL` | Judge model. Falls back to `MONOCLE_TEST_AGENT_MODEL`. |
| `MONOCLE_TRACE_TEST_AGENT` | `true` stops suppressing the test agent's spans, for debugging the tester. |

---

## Writing scenarios that test something

**Make the success criteria observable.** "Flight is booked" is judged against what the
agent *said*. The judge is instructed to be strict — a promise to book, or a follow-up
question, does not count as booked — so write criteria that a response can actually
demonstrate.

**Withhold what you want the agent to ask for.** A scenario where every param is
`is_initial` tests one long instruction, not a conversation. The interesting behavior
lives in what the agent does when something is missing.

**Let the persona do work.** A terse user who answers in three words, a user who changes
their mind, a user who is rude — these are the conditions under which agents drop context
or lose the thread. A polite, cooperative persona tests very little.

**Set `max_turns` to slightly more than the conversation should need.** It is a failure
budget, not a target. Raising it to make a failing scenario pass hides a loop that isn't
converging — read the transcript instead.

**Trace assertions and the judge answer different questions.** The judge reads what the
agent *said*; `monocle_trace_asserter` sees what it *did*. "Flight is booked" plus
`called_tool("book_flight")` is a much stronger test than either alone — an agent can
claim a booking without ever calling the tool.

---

## Debugging a failure

Start with `result.report()`. Each tool call gets its own line showing what it returned,
above the message that actually went to the target:

```
  turn 3  tester> [called source()] ==> San Francisco
          tester> [called destination()] ==> Seattle
          tester> I want to go to Seattle, from San Francisco.
          target> Which date would you like to travel?
          verdict> not met -- No booking confirmed.
```

The tester line is verbatim -- nothing is prepended to it -- so you can see exactly what
the target received. Between the fetched values, the message, and the judge's reasoning
per turn, this is usually enough to see whether the target went off task, the criteria
were unreachable, or the tester misbehaved.

If the tester itself looks wrong, set `MONOCLE_TRACE_TEST_AGENT=true` to stop suppressing
its spans, and its own inferences will show up in the traces alongside the target's.

Two failure shapes worth recognizing:

- **"max turns exhausted" with the target asking the same thing repeatedly** — the test
  agent isn't answering. Check that the param `description` matches the words the target
  uses to ask for it.
- **"required params never requested" while the judge says the criteria were met** — a
  detail was invented. Determine which side invented it: `param_tools_called` in the
  transcript tells you whether the tester ever fetched the real value.
