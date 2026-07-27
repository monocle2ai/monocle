# Monocle Coding Standards

> [!NOTE]
> These standards apply to all contributions in `apptrace/` and `test_tools/`. Follow them before raising a PR. See [[CONTRIBUTING]] for the full contribution workflow.

---

## Python Style

> [!NOTE]
> Monocle targets Python 3.8+. Avoid syntax or stdlib features unavailable on 3.8.

- **PEP 8** — 4-space indent, 120-char line limit (not 79).
- **Type hints** — all public functions must have full type annotations (`Optional`, `List`, `Dict` from `typing` for 3.8 compatibility).
- **Imports** — stdlib → third-party → local, separated by blank lines. No wildcard imports.
- **Docstrings** — one-line summary for public functions/classes. Multi-line only when the *why* is non-obvious. Never describe what the code does if the name already says it.
- **Comments** — write only when the *why* is non-obvious: hidden constraints, subtle invariants, workarounds for specific bugs. Delete comments that describe the obvious.

```python
# Good
def monocle_trace_method(span_name: Optional[str] = None, enabled: bool = True):
    """Decorator that wraps a function in a Monocle span.

    Args:
        span_name: Custom span name; defaults to the function name.
        enabled: When False the decorator is a passthrough — no spans created.
    """
```

---

## Span & Trace Naming Conventions

> [!WARNING]
> Do not invent new `span.type` values. Use the canonical set below. Incorrect span types break downstream eval pipelines.

| `span.type` value          | When to use                              |
|---------------------------|------------------------------------------|
| `agentic.invocation`      | Agent function call (invoke/stream)      |
| `agentic.tool.invocation` | Tool call inside an agent                |
| `agentic.turn`            | Full user-to-response turn               |
| `inference`               | Direct LLM inference                     |
| `inference.framework`     | LLM inference via a framework wrapper    |
| `workflow`                | Top-level workflow span                  |

**Attribute naming:** `entity.N.name` / `entity.N.type` (1-indexed). Never skip indices.

---

## Test Structure

> [!NOTE]
> Every code change must ship with a test. PRs without tests will not be merged.

### Unit tests

- Location: `apptrace/tests/unit/` or `test_tools/tests/unit/`
- Use `unittest.TestCase` for stateful test classes (span exporters). Use plain `pytest` functions for stateless checks.
- Mock external calls (subprocess, network) with `unittest.mock.patch`.
- Test file name: `test_<module_name>.py` matching the source file.

### Integration tests

- Location: `tests/` (framework-specific subdirs).
- May use real LLM calls gated behind env vars (`OPENAI_API_KEY`, etc.). Skip with `pytest.skip()` if the var is absent.

### Fluent API tests

Extend `test_tools/tests/unit/test_fluent_apis.py`. Always test both the positive path and the negative path (`does_not_have_attribute`, `does_not_contain_output`, etc.):

```python
def test_my_assertion(monocle_trace_asserter: TraceAssertion):
    monocle_trace_asserter.with_trace_source("file", trace_path="traces/trace1.json")
    monocle_trace_asserter.called_tool("my_tool") \
        .has_attribute("entity.1.type", "tool.langgraph") \
        .does_not_have_attribute("entity.1.type", "tool.openai")
```

### Generator tests

When extending `TestGenerator`, always add:

1. A unit test that verifies `analyze()` populates the new field.
2. A unit test that verifies `generate_test_code()` emits the expected assertion string.

---

## Fluent API Extension Pattern

> [!NOTE]
> All new assertion methods must be chainable — return `self` or a new `TraceAssertion`-compatible object.

```python
# In fluent_api.py
def has_my_assertion(self, expected: str) -> "TraceAssertion":
    """Assert X on the current span set."""
    # ... validate ...
    return self
```

Register new methods in `TraceAssertion.__all__` and add a corresponding entry in `test_generator.py`'s `generate_test_code()` so the generator can emit it automatically.

---

## Git Commit Standards

> [!WARNING]
> Every commit **must** be signed (`git commit -s`). Unsigned commits fail the DCO check and will block merge.

- Format: `type(scope): short summary` — e.g., `feat(fluent-api): add has_attribute to test generator`
- Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`
- Link the tracking issue in the PR body: `Fixes #NNN` or `Closes #NNN`
- Max 72 chars in subject line.
- Reference related issues inline: `# related to #690`

---

## PR Checklist

> [!NOTE]
> Complete this before requesting review.

- [ ] Code follows PEP 8 and type-hint rules above
- [ ] `span.type` values are from the canonical table
- [ ] Unit tests added/updated — no regressions
- [ ] All commits signed (`-s`)
- [ ] PR body references the tracking issue (`Fixes #NNN`)
- [ ] No debug prints or TODO comments left in production code
- [ ] `pylint` passes: `pylint src/` (configured via `.pylintrc`)

---

## Running Tests Locally

```bash
# apptrace unit tests
cd apptrace
python -m pytest tests/unit/ -x -q

# test_tools unit tests
cd test_tools
python -m pytest tests/unit/ -x -q

# Full matrix (requires tox)
tox
```

---

*Maintained by the Monocle committers. Propose changes via PR against `docs/CODING_STANDARDS.md`.*
