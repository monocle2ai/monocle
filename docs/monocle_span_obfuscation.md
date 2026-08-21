# Obfuscating sensitive data before export

Span `data.input` / `data.output` events carry the raw prompts, completions, tool arguments and
retrieved documents of your application, so they can contain API keys, passwords, PCI data or PII.
Monocle scrubs those payloads inside your process, before the span reaches any exporter.

**On by default**, redacting credentials — API keys, passwords, tokens and private keys — and
nothing else.

```python
@monocle_trace_method(span_name="call_model")
def call_model(prompt, config):
    return "ok"

call_model("capital of France?", {"api_key": openai_key, "user": "bob@corp.com"})
```

Your application still sees the real values; the exported `data.input` does not:

```json
{"input": "{\"args\": [\"capital of France?\", {\"api_key\": \"<API_KEY>\", \"user\": \"bob@corp.com\"}]}"}
```

`bob@corp.com` survives: the built-in obfuscator targets credentials, not PII. Names, addresses and
national IDs need context-aware detection rather than regexes — see
[Adding PII detection](#adding-pii-detection).

## Configuration

| Setting | Description |
| --- | --- |
| `MONOCLE_SPAN_OBFUSCATORS` | Comma-separated obfuscators, each a built-in name (`credentials`, `regex` as an alias for it, `presidio`) or an import path (`my_pkg.my_module:MyObfuscator`). Defaults to `credentials`. Set to `none`/`off`/`false`/`no`/`0`/`disabled` to turn obfuscation off. |
| `MONOCLE_DISABLE_SPAN_OBFUSCATION` | `true`/`1`/`yes`/`on` disables obfuscation regardless of the above. |
| `MONOCLE_OBFUSCATE_SPAN_TYPES` | Span types the env-configured obfuscators apply to, e.g. `inference,inference.*`. Defaults to all. Supports `*` wildcards. |
| `span_obfuscators=` | `setup_monocle_telemetry` argument. Wins over the environment. `[]` disables obfuscation; `None` (default) defers to the environment. |

Entries run in order, so passes can be layered. An unrecognized value for the disable flag leaves
obfuscation **on**, so a typo cannot silently drop protection. `set_span_obfuscators()` and
`register_span_obfuscator()` adjust the registry at runtime, which is useful in tests and in apps
that configure telemetry in stages.

## Built-in obfuscators

### `RegexSpanObfuscator`

The default. No extra packages, cheap enough to run on every span. Covers PEM private keys, JWTs,
bearer tokens, AWS/OpenAI/Anthropic/Google/GitHub/Slack keys, and a `credential_assignment`
catch-all that redacts the value of any `api_key`/`password`/`client_secret`-style assignment in
`key=value`, `key: value` and `"key": "value"` form — preserving the quotes, so a redacted JSON
payload stays parseable. `DEFAULT_PATTERNS` in
[span_obfuscator.py](../apptrace/src/monocle_apptrace/exporters/span_obfuscator.py) is the
authoritative list.

```python
RegexSpanObfuscator(
    patterns=["openai_api_key", "credential_assignment"],  # omit for all of them
    extra_patterns={"employee_id": (re.compile(r"\bEMP-\d{5}\b"), "<EMPLOYEE_ID>")},
    span_types=["inference", "inference.*"],
)
```

`extra_patterns` is the hook for organization-specific secret shapes. An unknown pattern name
raises rather than being ignored.

### Adding PII detection

`PresidioSpanObfuscator` wraps [Presidio](https://github.com/data-privacy-stack/presidio) for
NLP-based entity recognition (names, addresses, national IDs, card numbers) and richer operators
than replace — mask, hash, encrypt. Its models are much slower than the regex pass, so scope it
with `span_types`.

```bash
pip install monocle_apptrace[obfuscation]
# list both: entries replace the default rather than adding to it
export MONOCLE_SPAN_OBFUSCATORS=credentials,presidio
```

```python
PresidioSpanObfuscator(
    entities=["EMAIL_ADDRESS", "CREDIT_CARD", "PERSON", "US_SSN"],
    score_threshold=0.6,
    operators={"CREDIT_CARD": OperatorConfig("mask", {
        "masking_char": "*", "chars_to_mask": 12, "from_end": False,
    })},
    span_types=["inference", "inference.*"],
)
```

Pass a pre-built `analyzer=` to register your own recognizers.

## Writing your own obfuscator

Subclass `TextSpanObfuscator` to rewrite strings — it walks the payload, including strings nested in
lists, tuples and dicts, and hands you each one:

```python
class TenantSecretObfuscator(TextSpanObfuscator):
    span_types = ("inference", "inference.*")

    def obfuscate_text(self, text, key, event_name, span):
        return text.replace(secret_for(span.attributes.get("scope.tenant_id")), "<REDACTED>")
```

For full control over the payload dict — dropping keys, hashing, replacing it wholesale — subclass
`SpanObfuscator` and implement `obfuscate(payload, event_name, span)`; returning `{}` drops the
payload. Both hooks receive the span, so decisions can depend on span type, entity attributes or
scopes.

Register one by class path so it also works when telemetry is configured purely through the
environment. Obfuscators loaded this way are constructed with no arguments (plus `span_types=` when
`MONOCLE_OBFUSCATE_SPAN_TYPES` is set), so give them usable defaults.

```bash
export MONOCLE_SPAN_OBFUSCATORS=credentials,my_pkg.obfuscators:TenantSecretObfuscator
```

## Guarantees

- **Applied before export, at two layers.** `SpanExporterBase` routes every subclass's `export()`
  through `obfuscate_spans()`, so all Monocle exporters — including ones added later — scrub without
  any per-exporter code. Monocle also patches `on_end` on each span processor it installs, which
  covers custom `span_processors` and third-party exporters. Both layers are idempotent, so a span
  crossing both is scrubbed once. For an exporter outside both paths, use
  `wrap_exporter_with_obfuscation()`.
- **Spans are never mutated.** A shallow copy with rewritten events is exported. Obfuscation is
  idempotent, so a span crossing two obfuscation points is processed once.
- **Only the configured events change.** Span name, attributes, status, timings and other events
  such as `metadata` are exported unmodified.
- **Failures fail closed.** If an obfuscator raises or returns a non-dict, the payload it was handed
  is dropped rather than exported raw, and the rest of the span still exports.
- **Misconfiguration does not break the app.** An unknown obfuscator name or failing import logs a
  warning and is skipped.

Obfuscation defends against sensitive data reaching your trace store; it is not a substitute for
keeping secrets out of prompts. To drop a span type entirely use
[`SpanFilter`](../apptrace/src/monocle_apptrace/exporters/span_filter.py), or skip capture with
`@monocle_trace_method(exclude=["inputs", "outputs"])`.
