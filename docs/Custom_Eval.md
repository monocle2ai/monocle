# Custom Non-LLM Evaluations

This document describes the **non-LLM (deterministic) evaluators** built into
`monocle_test_tools`, and how to add your own.

Unlike the LLM-as-judge evaluators (`okahu`) or embedding-based scorers
(`bert_score`), non-LLM evals are **pure, deterministic functions** of a span's
input/output. They:

- **Require no API key and no network call** — no `OKAHU_API_KEY`, no model download.
- **Are fast and free** — they run in microseconds and add no token cost to CI.
- **Are fully reproducible** — the same input always yields the same score, so they
  never flake.

They are ideal for objective, rule-based checks: format/schema validity, presence of
required content, PII safety, exact-match correctness, readability budgets, and lexical
overlap against a reference answer.

## How non-LLM evals fit into the framework

Every evaluator is a subclass of
[`BaseEval`](../test_tools/src/monocle_test_tools/evals/base_eval.py). Its
`evaluate(eval_args: dict) -> dict` method receives the span data selected by the
evaluation's `args` (`input`, `output`, and/or `agent_description`) and returns a
**dict of metric → value**. A [comparer](monocle_test_assertions.md) then compares
that returned dict against your `expected_result`:

| Comparer key | Pass condition | Use for |
|--------------|----------------|---------|
| `metric`     | `actual[k] >= expected[k]` for every key `k` | numeric thresholds (scores, counts, coverage) |
| `default`    | `actual == expected` (exact dict equality)   | fixed/boolean outcomes |

Because all nine evaluators below return **numeric** values (typically `1.0`/`0.0`
flags plus raw scores), the `metric` comparer is the natural default: set
`expected_result` to the minimum acceptable value for each metric you care about.

The string keys are registered in
[`eval_manager.py`](../test_tools/src/monocle_test_tools/evals/eval_manager.py) under
`NON_LLM_EVALS`, so you can reference any of them by name from either the fluent API
(`with_evaluation("regex_match", {...})`) or the declarative `eval` block in a test
case (shown below).

---

## The nine built-in non-LLM evaluators

### 1. `regex_match` — `RegexMatchEval`
Checks whether the output matches a regular expression. Useful for asserting the
presence of a required format (an order ID, a URL, a date, a currency amount).

- **Inputs (`args`):** `output`
- **`eval_options`:**
  - `pattern` (str, **required**) — the regular expression.
  - `ignore_case` (bool, default `False`) — case-insensitive matching.
  - `full_match` (bool, default `False`) — require the whole output to match
    (`re.fullmatch`) rather than contain a match (`re.search`).
- **Returns:** `{"match": 1.0|0.0, "match_count": <float>}`

```python
eval = {
    "eval": "regex_match",
    "eval_options": {"pattern": r"order\s+#?\d{4,}"},
    "args": ["output"],
    "expected_result": {"match": 1.0},
    "comparer": "metric",
}
```

### 2. `json_validity` — `JSONValidityEval`
Verifies the output is well-formed JSON and, optionally, that it conforms to a JSON
schema (validated with `jsonschema`). Useful for testing structured/tool-style outputs.

- **Inputs (`args`):** `output`
- **`eval_options`:**
  - `json_schema` (dict, optional) — a JSON schema to validate the parsed output against.
    When omitted, only well-formedness is checked.
- **Returns:** `{"valid_json": 1.0|0.0, "schema_valid": 1.0|0.0}`
  (`schema_valid` is `1.0` when no schema was supplied and the JSON parses.)

```python
eval = {
    "eval": "json_validity",
    "eval_options": {"json_schema": {"type": "object", "required": ["id", "status"]}},
    "args": ["output"],
    "expected_result": {"valid_json": 1.0, "schema_valid": 1.0},
    "comparer": "metric",
}
```

### 3. `keyword_presence` — `KeywordPresenceEval`
Checks that required keywords appear and that forbidden keywords do **not** appear in
the output. Useful for content-coverage and simple safety/policy checks.

- **Inputs (`args`):** `output`
- **`eval_options`:**
  - `required_keywords` (list[str]) — keywords that should be present.
  - `forbidden_keywords` (list[str]) — keywords that must be absent.
  - `case_sensitive` (bool, default `False`).
- **Returns:**
  `{"required_coverage": <0.0-1.0>, "forbidden_absent": 1.0|0.0,
    "missing_required": <float>, "forbidden_found": <float>}`

```python
eval = {
    "eval": "keyword_presence",
    "eval_options": {
        "required_keywords": ["confirmation", "booking"],
        "forbidden_keywords": ["error", "sorry"],
    },
    "args": ["output"],
    "expected_result": {"required_coverage": 1.0, "forbidden_absent": 1.0},
    "comparer": "metric",
}
```

### 4. `exact_match` — `ExactMatchEval`
Deterministic exact-match between a reference (`input`) and the `output`, after optional
normalization. Useful for classification labels and canonical ground-truth answers.

- **Inputs (`args`):** `input`, `output`
- **`eval_options`:**
  - `normalize_whitespace` (bool, default `True`) — collapse runs of whitespace and strip.
  - `ignore_case` (bool, default `True`).
  - `ignore_punctuation` (bool, default `False`).
- **Returns:** `{"exact_match": 1.0|0.0}`

```python
eval = {
    "eval": "exact_match",
    "args": ["input", "output"],
    "expected_result": {"exact_match": 1.0},
    "comparer": "metric",
}
```

### 5. `pii_detection` — `PIIDetectionEval`
Scans the output for personally identifiable information using deterministic regular
expressions (emails, phone numbers, SSNs, credit-card numbers, IPv4 addresses). Useful
as a privacy/safety gate that should never leak PII.

- **Inputs (`args`):** `output`
- **`eval_options`:**
  - `pii_types` (list[str], optional) — subset of detectors to run
    (`email`, `phone`, `ssn`, `credit_card`, `ipv4`). Defaults to all.
  - `custom_patterns` (dict[str, str], optional) — extra `{name: regex}` detectors.
- **Returns:**
  `{"pii_free": 1.0|0.0, "pii_count": <float>, "pii_breakdown": {<type>: <float>, ...}}`

```python
eval = {
    "eval": "pii_detection",
    "args": ["output"],
    "expected_result": {"pii_free": 1.0},   # fail if any PII is found
    "comparer": "metric",
}
```

### 6. `readability` — `ReadabilityEval`
Computes deterministic readability metrics for the output — the **Flesch Reading Ease**
score and the **Flesch-Kincaid Grade Level** — from word/sentence/syllable counts.
Useful for enforcing that user-facing responses stay easy to read.

- **Inputs (`args`):** `output`
- **`eval_options`:** none
- **Returns:**
  `{"flesch_reading_ease": <float>, "flesch_kincaid_grade": <float>,
    "word_count": <float>, "sentence_count": <float>}`
  (Higher reading-ease = easier; for grade level, *lower* is easier — invert the check
  accordingly, e.g. assert a reading-ease floor.)

```python
eval = {
    "eval": "readability",
    "args": ["output"],
    "expected_result": {"flesch_reading_ease": 50.0},  # require reading ease >= 50
    "comparer": "metric",
}
```

### 7. `token_overlap` — `TokenOverlapEval`
Measures ROUGE-style lexical overlap (precision / recall / F1) between a reference
(`input`) and the `output` as a bag of tokens. A deterministic alternative to
embedding-based similarity when you want a no-dependency, no-model relevance check.

- **Inputs (`args`):** `input`, `output`
- **`eval_options`:**
  - `ignore_case` (bool, default `True`).
  - `token_pattern` (str, default `r"[A-Za-z0-9']+"`) — regex used to extract tokens.
- **Returns:** `{"precision": <0.0-1.0>, "recall": <0.0-1.0>, "f1": <0.0-1.0>}`

```python
eval = {
    "eval": "token_overlap",
    "args": ["input", "output"],
    "expected_result": {"f1": 0.6},   # require F1 overlap >= 0.6
    "comparer": "metric",
}
```

### 8. `bleu` — `BleuEval`
Computes sentence-level **BLEU** between a reference (`input`) and a candidate
(`output`) — the standard precision-based n-gram overlap metric with a brevity
penalty. Pure Python; no model or external dependency. Useful for translation/
generation fidelity against a gold reference.

- **Inputs (`args`):** `input`, `output`
- **`eval_options`:**
  - `max_n` (int, default `4`) — maximum n-gram order (BLEU-N).
  - `weights` (list[float], optional) — per-order weights; defaults to uniform.
  - `smooth` (bool, default `True`) — epsilon-smooth zero-count orders.
  - `ignore_case` (bool, default `True`), `token_pattern` (str).
- **Returns:** `{"bleu": <0.0-1.0>, "brevity_penalty": <float>, "precision_1": <float>, ...}`

```python
eval = {
    "eval": "bleu",
    "eval_options": {"max_n": 4},
    "args": ["input", "output"],
    "expected_result": {"bleu": 0.3},   # require BLEU >= 0.3
    "comparer": "metric",
}
```

### 9. `rouge` — `RougeEval`
Computes **ROUGE** recall-oriented overlap between a reference (`input`) and a
candidate (`output`): ROUGE-N (n-gram overlap) and ROUGE-L (longest common
subsequence), each with precision/recall/F1. Pure Python; no model or external
dependency. The standard metric family for summarization quality.

- **Inputs (`args`):** `input`, `output`
- **`eval_options`:**
  - `rouge_types` (list[str], default `["rouge1", "rouge2", "rougeL"]`) — any of
    `rouge1`, `rouge2`, … and `rougeL`.
  - `ignore_case` (bool, default `True`), `token_pattern` (str).
- **Returns (flat, numeric — one set per type):**
  `{"rouge1_p": <float>, "rouge1_r": <float>, "rouge1_f": <float>, "rougeL_f": <float>, ...}`

```python
eval = {
    "eval": "rouge",
    "eval_options": {"rouge_types": ["rouge1", "rougeL"]},
    "args": ["input", "output"],
    "expected_result": {"rougeL_f": 0.4},   # require ROUGE-L F1 >= 0.4
    "comparer": "metric",
}
```

---

## Using non-LLM evals in a test case

Non-LLM evals plug into the declarative `eval` block of a `TestSpan` exactly like the
built-in `bert_score` evaluator. The example below attaches a `json_validity` and a
`pii_detection` check to the final agentic turn:

```python
import pytest
from monocle_test_tools import MonocleValidator
from test_common.adk_travel_agent import root_agent

agent_test_cases = [
    {
        "test_input": ["Book a flight from San Jose to Seattle for 27th Nov 2025."],
        "test_spans": [
            {
                "span_type": "agentic.turn",
                "eval": {
                    "eval": "pii_detection",
                    "args": ["output"],
                    "expected_result": {"pii_free": 1.0},
                    "comparer": "metric",
                },
            },
        ],
    },
]

@pytest.mark.asyncio
@pytest.mark.parametrize("monocle_test_case", agent_test_cases)
async def test_run_agents(monocle_test_case):
    await MonocleValidator().test_agent_async(root_agent, "google_adk", monocle_test_case)
```

You can also construct an evaluator directly in Python:

```python
from monocle_test_tools.evals import get_evaluator

evaluator = get_evaluator("token_overlap", {"ignore_case": True})
score = evaluator.evaluate({"input": "the quick brown fox", "output": "the brown fox jumped"})
# -> {"precision": 0.75, "recall": 0.75, "f1": 0.75}
```

## Running the end-to-end test cases

The repository includes a full non-LLM eval test path in the unit test file
[`test_non_llm_evals.py`](../test_tools/tests/unit/test_non_llm_evals.py) and a
live end-to-end smoke test in the integration test file
[`test_non_llm_evals.py`](../test_tools/tests/integration/test_non_llm_evals.py).

The simplest local setup is to run them from a clean virtual environment and call
`pytest` through the environment's Python, not the shell's global `pytest` entrypoint.

```bash
cd monocle
python3 -m venv .venv-py310
./.venv-py310/bin/python -m pip install -U pip
./.venv-py310/bin/python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  requests wrapt rfc3986 pyyaml opentelemetry-api==1.42.1 opentelemetry-sdk==1.42.1 \
  opentelemetry-instrumentation==0.63b1 opentelemetry-exporter-otlp-proto-http==1.42.1 \
  pydantic==2.11.7 jsonschema==4.23.0 pytest==8.3.5 pytest-asyncio==0.26.0 \
  GitPython==3.1.45 bert-score==0.3.13 transformers==4.57.3 sentence-transformers==3.3.1 \
  google-adk==1.10.0 google-generativeai==0.8.5 editables
./.venv-py310/bin/python -m pip install --no-deps --no-build-isolation -e ./apptrace -e ./test_tools
```

Run the deterministic unit suite first. This verifies every built-in non-LLM
evaluator and the `NON_LLM_EVALS` registration table.

```bash
HOME="$PWD" GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv-py310/bin/pytest -p pytest_asyncio.plugin \
  test_tools/tests/unit/test_non_llm_evals.py
```

Run the live end-to-end smoke test after the unit suite passes. This exercises the
full flow: ADK agent run -> trace capture -> span extraction -> deterministic evals
-> comparer checks.

```bash
export GOOGLE_API_KEY=your_google_api_key
# or, if you use Vertex AI:
# export GOOGLE_GENAI_USE_VERTEXAI=true
# export GOOGLE_CLOUD_PROJECT=your_project
# export GOOGLE_CLOUD_LOCATION=your_location

HOME="$PWD" GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv-py310/bin/pytest -p pytest_asyncio.plugin \
  test_tools/tests/integration/test_non_llm_evals.py -k all_non_llm_evals_on_single_run
```

Notes:

- The unit suite should pass without any Google credentials.
- The integration smoke test will fail if `GOOGLE_API_KEY` is empty and Vertex AI
  settings are not configured.
- In this workspace, `pytest` from the shell can resolve to a different Python
  installation, so prefer the explicit `./.venv-py310/bin/pytest` command above.

---

## Writing your own non-LLM eval

Adding a new deterministic evaluator follows the same three-step pattern used by the
built-ins:

1. **Create the evaluator** under
   `test_tools/src/monocle_test_tools/evals/<name>_eval.py`. Subclass `BaseEval`,
   read any configuration from `self.eval_options` in `__init__`, and implement
   `evaluate(eval_args: dict) -> dict`. Return a dict of numeric metrics so it works
   with the `metric` comparer.

   ```python
   from monocle_test_tools.evals.base_eval import BaseEval
   from pydantic import Field

   class WordCountEval(BaseEval):
       """Fails if the output exceeds a maximum word count."""
       max_words: int = Field(default=200, description="Maximum allowed word count.")

       def __init__(self, **data):
           super().__init__(**data)
           self.max_words = self.eval_options.get("max_words", self.max_words)

       def evaluate(self, eval_args: dict) -> dict:
           output = eval_args.get("output") or ""
           count = len(str(output).split())
           return {"within_limit": 1.0 if count <= self.max_words else 0.0,
                   "word_count": float(count)}
   ```

2. **Register a string key** in
   [`eval_manager.py`](../test_tools/src/monocle_test_tools/evals/eval_manager.py) by
   adding it to the `NON_LLM_EVALS` mapping, and **export it** from
   [`evals/__init__.py`](../test_tools/src/monocle_test_tools/evals/__init__.py).

   ```python
   # eval_manager.py
   from monocle_test_tools.evals.word_count_eval import WordCountEval
   NON_LLM_EVALS = {
       # ... existing entries ...
       "word_count": WordCountEval,
   }
   ```

3. **Use it** by name in a test, just like the built-ins:

   ```python
   "eval": {
       "eval": "word_count",
       "eval_options": {"max_words": 150},
       "args": ["output"],
       "expected_result": {"within_limit": 1.0},
       "comparer": "metric",
   }
   ```

### Guidelines

- Keep evaluators **pure and deterministic** — no network calls, no randomness, no
  wall-clock dependence.
- Accept input via the standard `eval_args` keys (`input`, `output`,
  `agent_description`) and declare which you need through the test's `args` list.
- Prefer **numeric** return values so the `metric` comparer can apply thresholds; use
  `1.0`/`0.0` for boolean-style flags.
- Override `cleanup()` only if your evaluator holds resources that must be released at
  the end of a test run (none of the built-in non-LLM evals do).

## See also

- [monocle_evaluation_api.md](monocle_evaluation_api.md) — the LLM-based (`okahu`)
  evaluation API and the `monocle_trace_asserter` fixture.
- [monocle_test_assertions.md](monocle_test_assertions.md) — comparers and assertion
  semantics.
- [evaluation_error_examples.md](evaluation_error_examples.md) — interpreting
  evaluation failures.
