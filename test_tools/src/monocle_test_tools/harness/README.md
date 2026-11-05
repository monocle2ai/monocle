# Scenario tests: how they work

The [test_tools README](../../../README.md#scenario-tests-monocle_scenario_testcase) shows how
to run them. This page covers how they work and the options.

## From a seed to scenarios

On the first run, a model splits each goal into its details (`source=Bombay`,
`date=22 Oct 2026`) and writes what success looks like. Each goal then becomes:

- one scenario per persona in [`personas.yaml`](../scenarios/personas.yaml)
- one per detail the user holds back until asked, plus one holding back all but the first
- one asking for something the agent can't do
- red-team attacks from [`redteam.yaml`](../scenarios/redteam.yaml): getting the agent to act
  without a detail, to reveal its instructions, or, when the goal has it read an email, page
  or document, to obey instructions hidden in that content. Where a
  [DeepTeam](https://github.com/confident-ai/deepteam) template fits, it rewrites the attack.

Each `out_of_scope` entry adds a scenario asking for that thing. Everything is saved to
`scenarios.json` and reused until the seed changes.

## One conversation

A model plays the user in the scenario's persona. Each held-back detail is a tool it can call
only when your agent asks for it, so it can't leak early. After every reply, a judge checks
the success criteria against what your agent said and the tools it called. The scenario
passes when the criteria are met, and fails when your agent breaks them, runs out of turns,
or finishes without asking for a held-back detail (it made one up).

The simulated user and the judge aren't traced, so the traces hold only your agent.

## Options

| Seed key | Meaning |
|---|---|
| `agent` | What the agent is. |
| `goals` | What people use it for, typed as a user would, with real values. |
| `out_of_scope` | Things it should decline. Optional. |
| `max_turns` | Turn limit per scenario. Default 8. |

**Writing the scenarios yourself.** The seed can instead be a list of scenarios, run as written:

```python
SEED = [{
    "test_name": "book_flight_impatient",
    "scenario": "Book a flight from source to destination.",
    "persona": "An impatient user who gives one detail at a time.",
    "params": [
        {"name": "source", "value": "San Francisco", "description": "source airport", "is_initial": True},
        {"name": "date", "value": "22 Oct 2026", "description": "travel date"},
    ],
    "success_criteria": "The agent confirms the flight is booked.",
    "max_turns": 6,
    "target_description": "A travel booking agent that books flights and hotels.",
}]
```

`params` without `is_initial` are held back until asked; `required: false` lets the agent skip
one. `min_turns` stops the criteria counting as met before that turn.

**Mock tools.** A scenario can carry `mock_tools`, in the same shape as `MockTool` (see the
[test_tools README](../../../README.md#mock-tools)). They apply on every turn of the conversation, so
the agent's tools never reach a real service:

```python
"mock_tools": [{"name": "search_flights", "type": "tool.adk",
                "response": [{"flight_id": "FL101", "to": "{{destination}}"}]},
               {"name": "book_flight", "type": "tool.adk",
                "response": {"status": "confirmed", "flight_id": "{{flight_id}}"}}],
```

**Models.** `test_scenario_async(..., model=, judge_model=)` takes a model object or a
`provider:model` string. Otherwise `MONOCLE_TEST_AGENT_MODEL` plays the user and
`MONOCLE_TEST_JUDGE_MODEL` (default: the same) judges. `MONOCLE_TRACE_TEST_AGENT=true` traces
the simulated user and the judge too, for debugging them.
