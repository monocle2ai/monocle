# Loading Traces for Monocle Pytest Assertions

This guide covers the two file-based methods for loading pre-existing traces into the Monocle pytest assertion framework: loading from a local JSON file and loading from the Okahu service.

For live agent execution via `run_agent_async()`, see [monocle_evaluation_api.md](monocle_evaluation_api.md).

## When to Use Each Method

| | JSON File | Okahu Service | Live Agent (`run_agent_async`) |
|---|---|---|---|
| **Source** | Local `.json` trace file | Okahu REST API | Runs the agent in-process |
| **Network** | None for structural tests; Okahu for evals | Always (API key required) | Agent deps + Okahu for evals |
| **Speed** | Fast (local I/O only) | Network latency per call | Full agent execution time |
| **Best for** | CI pipelines, offline dev, recorded regression traces | Validating production traces, traces already in Okahu | End-to-end integration tests |
| **Trace must exist** | As a local file | In Okahu (already ingested) | No (generated on the fly) |

**Choose JSON** when you have exported trace files and want fast, deterministic, offline-capable tests. Structural assertions (agent/tool/input/output checks) need no network at all.

**Choose Okahu** when you want to validate traces that already exist in the Okahu service without re-exporting them.

## Loading from JSON

### Setup

```bash
pip install monocle_test_tools
```

No `OKAHU_API_KEY` is needed for structural-only tests. If you also run `check_eval()`, you will need it.

### Trace File Format

The JSON trace file is a Monocle trace export — an array of span objects. Each span contains attributes like `span.type`, `entity.name`, input/output events, and timing data. These files are produced by the Monocle exporter when `MONOCLE_EXPORTER` includes `file`.

### Fixture

The fixture sets `_trace_source = "local"`. This tells the framework that spans are not yet on Okahu, so if you call `check_eval()`, it will:
1. Export the spans to Okahu before running the eval
2. Delete them from Okahu during cleanup

```python
import pytest
from monocle_test_tools import TraceAssertion

@pytest.fixture()
def monocle_trace_asserter():
    asserter = TraceAssertion()
    asserter.cleanup()
    asserter.validator._trace_source = "local"
    yield asserter
    asserter.cleanup()
```

### Loading Spans

```python
import os
from monocle_test_tools.file_span_loader import JSONSpanLoader

TRACE_PATH = os.path.join(os.path.dirname(__file__), "traces", "trace1.json")

def test_structural_only(monocle_trace_asserter):
    monocle_trace_asserter.load_spans(JSONSpanLoader.from_json(TRACE_PATH))

    monocle_trace_asserter.called_tool("book_hotel", "booking_agent") \
        .contains_input("Mumbai") \
        .contains_output("Successfully booked")
```

### Two-Tier Pattern

JSON tests can be split into two tiers: structural-only (offline, free) and structural + eval (needs Okahu):

```python
# Tier 1: Offline, no API key needed
def test_agent_path(monocle_trace_asserter):
    monocle_trace_asserter.load_spans(JSONSpanLoader.from_json(TRACE_PATH))
    monocle_trace_asserter.called_agent("booking_agent") \
        .contains_output("booked")

# Tier 2: Structural guard + Okahu eval
@pytest.mark.eval
def test_agent_with_eval(monocle_trace_asserter):
    monocle_trace_asserter.load_spans(JSONSpanLoader.from_json(TRACE_PATH))

    # Structural guard runs first (fast, free)
    monocle_trace_asserter.called_agent("booking_agent") \
        .contains_output("booked")

    # Eval runs only if the guard passes (slow, paid)
    monocle_trace_asserter.with_evaluation("okahu") \
        .check_eval("hallucination", expected="no_hallucination")
```

Run tiers selectively:
```bash
pytest test_file.py -m "not eval"   # structural only (offline)
pytest test_file.py -m eval         # eval tests only
```

### Example File

See `test_json_trace_assertions.py` for a complete working example.

---

## Loading from Okahu

### Setup

```bash
pip install monocle_test_tools
```

Required environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OKAHU_API_KEY` | Yes | Your Okahu API key |

### Fixture

The fixture sets `_trace_source = "okahu"`. This tells the framework that spans already exist on Okahu, so:
- `check_eval()` skips the export step (no re-ingestion)
- Cleanup does not delete the trace from Okahu

```python
import pytest
from monocle_test_tools import TraceAssertion

@pytest.fixture()
def monocle_trace_asserter():
    asserter = TraceAssertion()
    asserter.cleanup()
    asserter.validator._trace_source = "okahu"
    yield asserter
    asserter.cleanup()
```

### Loading Spans

You need two values: the **workflow name** (the service/workflow registered in Okahu) and the **trace ID** (a hex string identifying the trace).

```python
from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

WORKFLOW = "my_workflow_name"
TRACE_ID = "642dbd9d0dfcfdbdc8849f67f34c8a19"

def test_from_okahu(monocle_trace_asserter):
    monocle_trace_asserter.load_spans(
        OkahuSpanLoader.get_spans(WORKFLOW, TRACE_ID)
    )

    monocle_trace_asserter.called_agent("my_agent") \
        .contains_input("Mumbai") \
        .contains_output("booked")
```

### Additional Okahu Loading Methods

`OkahuSpanLoader` also supports loading by session or scope:

```python
# Load all spans in a session
OkahuSpanLoader.load_by_session(workflow_name, session_id="session_123")

# Load spans by scope
OkahuSpanLoader.load_by_scope(workflow_name, scope_key="key", scope_value="value")

# Get trace IDs matching a fact
OkahuSpanLoader.get_trace_ids(workflow_name, fact_name="agentic_session", fact_id="session_123")
```

### Example File

See `test_okahu_trace_assertions.py` for a complete working example.

---

## Key Differences at a Glance

| Behavior | JSON (`_trace_source = "local"`) | Okahu (`_trace_source = "okahu"`) |
|----------|---|---|
| Spans exported before eval | Yes | No (already on Okahu) |
| Trace deleted on cleanup | Yes | No |
| `shadow_eval` flag | `True` | `False` |
| Network needed for structural tests | No | Yes |
| Ingest delay needed | No | No |

## Related Documentation

- [monocle_test_assertions.md](monocle_test_assertions.md) — Full assertion API reference
- [monocle_evaluation_api.md](monocle_evaluation_api.md) — Eval metrics, fact names, check_eval syntax, performance testing
- [evaluation_error_examples.md](evaluation_error_examples.md) — Common eval error patterns and debugging
