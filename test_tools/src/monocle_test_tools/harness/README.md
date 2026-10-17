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

**Models.** `test_scenario_async(..., model=, judge_model=)` takes a model object, a
`provider:model` string, or a local model below. Otherwise `MONOCLE_TEST_AGENT_MODEL` plays the
user and `MONOCLE_TEST_JUDGE_MODEL` (default: the same) judges. `MONOCLE_TRACE_TEST_AGENT=true`
traces the simulated user and the judge too, for debugging them.

## Local models

For when there's no API key to use: `model="local"` runs a small open model on your machine.
The first time, test_tools installs `llama-cpp-python` from a prebuilt wheel and downloads the
model (about 2.5 GB). The simulated user and the judge must use the same local model.

| `model=` | Model | Made by | Same result as gpt-4.1-mini | Time per scenario |
|---|---|---|---|---|
| `local` | Qwen3-4B-Instruct-2507 | Alibaba (Qwen) | 7 of 8 | 21 s |
| `local:llama-3.2-3b` | Llama 3.2 3B Instruct | Meta | 6 of 8 | 16 s |
| `local:phi-4-mini` | Phi-4-mini-instruct | Microsoft | 4 of 8 | 15 s |

Measured on eight scenarios against the example travel agent, each model playing the user and
judging, on an M4 MacBook's GPU; each needs about 4 to 5 GB of memory. Qwen3-4B is the default
because it handled held-back details best. Llama 3.2 3B was close and slipped on two red-team
scenarios. Phi-4-mini often missed held-back details and misjudged a red-team scenario. On a CPU
alone they work, but a scenario takes minutes. A hosted model is faster and more reliable.

Small models are weaker at writing the scenarios than at running them: Qwen3-4B, for one, may
not notice that a goal has the agent read an email or document, so the injection attacks for it
aren't generated. For the full set, generate once with a hosted model on the decorator
(`monocle_scenario_testcase(SEED, model="openai:gpt-4.1-mini")`) and run the conversations with
`model="local"`; the scenarios are cached, so that's the only call that needs a key.
