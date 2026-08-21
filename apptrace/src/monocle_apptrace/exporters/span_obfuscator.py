"""
Scrubs sensitive data from span ``data.input`` / ``data.output`` payloads before
the span reaches any exporter.

On by default, redacting credentials -- API keys, passwords, tokens and private
keys. For PII (names, addresses, national IDs) add the Presidio obfuscator,
which needs the optional ``obfuscation`` extra.

    MONOCLE_DISABLE_SPAN_OBFUSCATION=true         # off
    MONOCLE_SPAN_OBFUSCATORS=credentials,presidio # add PII detection

An obfuscator is any subclass of :class:`SpanObfuscator`; subclass
:class:`TextSpanObfuscator` to rewrite strings, :class:`SpanObfuscator` for full
control over the payload dict.

See docs/monocle_span_obfuscation.md for configuration and examples.
"""

import copy
import logging
import os
import re
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExportResult

from monocle_apptrace.instrumentation.common.constants import (
    DATA_INPUT_KEY,
    DATA_OUTPUT_KEY,
    SPAN_TYPE_KEY,
)

logger = logging.getLogger(__name__)

SPAN_OBFUSCATORS_ENV = "MONOCLE_SPAN_OBFUSCATORS"
OBFUSCATE_SPAN_TYPES_ENV = "MONOCLE_OBFUSCATE_SPAN_TYPES"
DISABLE_OBFUSCATION_ENV = "MONOCLE_DISABLE_SPAN_OBFUSCATION"

OBFUSCATION_OFF_VALUES = ("none", "off", "false", "no", "0", "disabled")

DEFAULT_OBFUSCATED_EVENTS = (DATA_INPUT_KEY, DATA_OUTPUT_KEY)

_OBFUSCATED_MARKER = "_monocle_obfuscated"

_OBFUSCATION_HOOK_MARKER = "_monocle_obfuscation_hook"


def matches_span_type(span_type: str, pattern: str) -> bool:
    """Match a span type against a pattern.

    ``"*"`` matches everything, ``"inference.*"`` anything starting with
    ``"inference"``, ``"*.framework"`` anything ending in ``".framework"``.
    """
    if pattern == "*":
        return True
    if "*" in pattern:
        if pattern.endswith(".*"):
            return span_type.startswith(pattern[:-2])
        if pattern.startswith("*."):
            return span_type.endswith(pattern[2:])
        regex = pattern.replace(".", r"\.").replace("*", ".*")
        return re.match(f"^{regex}$", span_type) is not None
    return span_type == pattern


class SpanObfuscator(ABC):
    """Base class for pluggable span payload obfuscators."""

    span_types: Sequence[str] = ("*",)

    event_names: Sequence[str] = DEFAULT_OBFUSCATED_EVENTS

    def __init__(
        self,
        span_types: Optional[Sequence[str]] = None,
        event_names: Optional[Sequence[str]] = None,
    ) -> None:
        if span_types is not None:
            self.span_types = tuple(span_types)
        if event_names is not None:
            self.event_names = tuple(event_names)

    def applies_to(self, span: ReadableSpan) -> bool:
        """Return True if this obfuscator should run for *span*."""
        span_type = (getattr(span, "attributes", None) or {}).get(SPAN_TYPE_KEY, "") or ""
        return any(matches_span_type(span_type, pattern) for pattern in self.span_types)

    def applies_to_event(self, event_name: str) -> bool:
        """Return True if *event_name*'s payload should be obfuscated."""
        return event_name in self.event_names

    @abstractmethod
    def obfuscate(
        self, payload: Dict[str, Any], event_name: str, span: ReadableSpan
    ) -> Dict[str, Any]:
        """Return the payload to export in place of *payload*.

        *payload* is a mutable copy of the event's attributes. Returning an empty
        dict drops the payload. *span* is passed for context, e.g. span type or
        scope attributes.
        """
        raise NotImplementedError


class TextSpanObfuscator(SpanObfuscator):
    """Base class for obfuscators that only rewrite text.

    Walks the payload -- including strings nested in lists, tuples and dicts --
    and passes every string through :meth:`obfuscate_text`.
    """

    @abstractmethod
    def obfuscate_text(
        self, text: str, key: str, event_name: str, span: ReadableSpan
    ) -> str:
        """Return the obfuscated replacement for one string, found under *key*."""
        raise NotImplementedError

    def obfuscate(
        self, payload: Dict[str, Any], event_name: str, span: ReadableSpan
    ) -> Dict[str, Any]:
        return {
            key: self._walk(value, key, event_name, span)
            for key, value in payload.items()
        }

    def _walk(self, value: Any, key: str, event_name: str, span: ReadableSpan) -> Any:
        if isinstance(value, str):
            return self.obfuscate_text(value, key, event_name, span)
        if isinstance(value, (list, tuple)):
            walked = [self._walk(item, key, event_name, span) for item in value]
            return tuple(walked) if isinstance(value, tuple) else walked
        if isinstance(value, dict):
            return {k: self._walk(v, key, event_name, span) for k, v in value.items()}
        return value


_PLACEHOLDER = re.compile(r"^<[A-Z_]+>$")

# Auth scheme keywords that sit between the key and the actual secret, as in
# "Authorization: Bearer <token>". The scheme is not the secret.
_AUTH_SCHEMES = frozenset({"bearer", "basic", "digest", "token", "apikey"})


def _redact_credential_assignment(match: "re.Match") -> str:
    """Redact the value of an ``api_key: ...`` style assignment.

    Preserves the quotes around the value so redacting a secret inside a JSON
    payload leaves the payload parseable, and leaves the match alone when a more
    specific pattern already replaced the value -- so
    ``Authorization: Bearer <TOKEN>`` keeps its precise marker instead of
    degrading to ``Authorization: <REDACTED>``.
    """
    prefix, value = match.group(1), match.group(2)
    bare = value.strip("\"'")
    if _PLACEHOLDER.match(bare) or bare.lower() in _AUTH_SCHEMES:
        return match.group(0)
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        quote = value[0]
        return f"{prefix}{quote}<REDACTED>{quote}"
    return f"{prefix}<REDACTED>"


# Ordered so that specific credential shapes are redacted before the generic
# "key = value" catch-all gets a chance to mangle them.
DEFAULT_PATTERNS: Dict[str, Any] = {
    "private_key": (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "<PRIVATE_KEY>",
    ),
    "jwt": (
        re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        "<JWT>",
    ),
    "bearer_token": (
        re.compile(r"\b[Bb]earer\s+[A-Za-z0-9\-._~+/]{8,}=*"),
        "Bearer <TOKEN>",
    ),
    "aws_access_key": (
        re.compile(
            r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16}\b"
        ),
        "<AWS_ACCESS_KEY>",
    ),
    "openai_api_key": (
        re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}\b"),
        "<API_KEY>",
    ),
    "google_api_key": (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "<API_KEY>"),
    "github_token": (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), "<API_KEY>"),
    "slack_token": (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "<API_KEY>"),
    # Generic "api_key": "value" / password=value assignments, including the
    # quoted form found in JSON payloads and the "x-api-key: value" header form.
    "credential_assignment": (
        re.compile(
            r"(?i)\b((?:api[_-]?key|apikey|secret|password|passwd|pwd|"
            r"access[_-]?token|auth[_-]?token|refresh[_-]?token|id[_-]?token|"
            r"bearer[_-]?token|session[_-]?token|sas[_-]?token|"
            r"client[_-]?secret|private[_-]?key|credentials?|authorization)"
            r"\b[\"']?\s*[:=]\s*)"
            r"(\"[^\"]+\"|'[^']+'|[^\s,;}\"']+)"
        ),
        _redact_credential_assignment,
    ),
}

#: All built-in patterns are credential patterns, and all are on by default.
CREDENTIAL_PATTERNS = tuple(DEFAULT_PATTERNS)
DEFAULT_ENABLED_PATTERNS = CREDENTIAL_PATTERNS


def _resolve_pattern_names(patterns: Any) -> tuple:
    """Validate a ``patterns`` argument into an ordered tuple of pattern names.

    Accepts ``None`` (all built-ins), a single pattern name, or a sequence of
    them. Order follows :data:`DEFAULT_PATTERNS` so specific credential shapes
    are always redacted before the generic ``key = value`` catch-all sees the
    text.
    """
    if patterns is None:
        return DEFAULT_ENABLED_PATTERNS
    names = (patterns,) if isinstance(patterns, str) else tuple(patterns)

    unknown = [name for name in names if name not in DEFAULT_PATTERNS]
    if unknown:
        raise ValueError(
            f"Unknown obfuscation pattern(s) {unknown}. "
            f"Known patterns: {sorted(DEFAULT_PATTERNS)}"
        )
    return tuple(name for name in DEFAULT_PATTERNS if name in names)


class RegexSpanObfuscator(TextSpanObfuscator):
    """Dependency-free obfuscator that redacts well-known credential shapes.

    Covers the shapes in :data:`DEFAULT_PATTERNS`, all enabled unless *patterns*
    narrows them. *extra_patterns* takes additional
    ``{name: (compiled_regex, replacement)}`` entries for organization-specific
    secrets, applied after the built-ins.
    """

    def __init__(
        self,
        patterns: Optional[Sequence[str]] = None,
        extra_patterns: Optional[Dict[str, Any]] = None,
        span_types: Optional[Sequence[str]] = None,
        event_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(span_types=span_types, event_names=event_names)
        self.pattern_names = _resolve_pattern_names(patterns)
        self.patterns: List[Any] = [
            DEFAULT_PATTERNS[name] for name in self.pattern_names
        ]
        if extra_patterns:
            self.patterns.extend(extra_patterns.values())

    def obfuscate_text(
        self, text: str, key: str, event_name: str, span: ReadableSpan
    ) -> str:
        for regex, replacement in self.patterns:
            text = regex.sub(replacement, text)
        return text


class PresidioSpanObfuscator(TextSpanObfuscator):
    """Obfuscator backed by the Presidio analyzer/anonymizer engines.

    Needs ``pip install monocle_apptrace[obfuscation]``. Its NLP models are much
    slower than :class:`RegexSpanObfuscator`, so scope it with ``span_types``.
    Pass pre-built ``analyzer``/``anonymizer`` engines to register custom
    recognizers, or ``operators`` to mask/hash/encrypt instead of replace.
    """

    def __init__(
        self,
        entities: Optional[Sequence[str]] = None,
        language: str = "en",
        score_threshold: float = 0.5,
        operators: Optional[Dict[str, Any]] = None,
        analyzer: Any = None,
        anonymizer: Any = None,
        span_types: Optional[Sequence[str]] = None,
        event_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(span_types=span_types, event_names=event_names)
        self.entities = list(entities) if entities else None
        self.language = language
        self.score_threshold = score_threshold
        self.operators = operators
        if analyzer is None or anonymizer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
            except ImportError as ex:
                raise ImportError(
                    "PresidioSpanObfuscator requires the presidio-analyzer and "
                    "presidio-anonymizer packages. Install them with "
                    "'pip install monocle_apptrace[obfuscation]'."
                ) from ex
            analyzer = analyzer or AnalyzerEngine()
            anonymizer = anonymizer or AnonymizerEngine()
        self.analyzer = analyzer
        self.anonymizer = anonymizer

    def obfuscate_text(
        self, text: str, key: str, event_name: str, span: ReadableSpan
    ) -> str:
        if not text.strip():
            return text
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language=self.language,
            score_threshold=self.score_threshold,
        )
        if not results:
            return text
        return self.anonymizer.anonymize(
            text=text, analyzer_results=results, operators=self.operators
        ).text


# ----------------------------------------------------------------------------
# Applying obfuscators to spans
# ----------------------------------------------------------------------------

def obfuscate_span(
    span: ReadableSpan, obfuscators: Sequence[SpanObfuscator]
) -> ReadableSpan:
    """Return *span* with its configured event payloads obfuscated.

    The input span is never mutated; when anything changes, a shallow copy with
    rewritten events is returned. Spans already obfuscated, spans no obfuscator
    applies to, and spans whose payloads come back unchanged are returned as-is.
    """
    if not obfuscators or getattr(span, _OBFUSCATED_MARKER, False):
        return span

    applicable = [obf for obf in obfuscators if obf.applies_to(span)]
    if not applicable:
        return span

    # Span-like objects reaching an exporter may not expose events (test doubles,
    # third-party wrappers). Nothing to scrub, and obfuscation must never break
    # an export.
    events = getattr(span, "events", None) or ()
    if not events:
        return span

    new_events = []
    changed = False
    for event in events:
        matching = [obf for obf in applicable if obf.applies_to_event(event.name)]
        if not matching:
            new_events.append(event)
            continue

        payload = dict(event.attributes or {})
        original = payload
        for obfuscator in matching:
            try:
                result = obfuscator.obfuscate(dict(payload), event.name, span)
            except Exception as ex:
                # A broken obfuscator must not leak the payload it failed on, and
                # must not take the export down either -- drop the payload.
                logger.warning(
                    "Span obfuscator %s failed on event '%s', dropping its payload: %s",
                    type(obfuscator).__name__, event.name, ex,
                )
                payload = {}
                break
            if not isinstance(result, dict):
                logger.warning(
                    "Span obfuscator %s returned %s for event '%s', expected dict; "
                    "dropping its payload.",
                    type(obfuscator).__name__, type(result).__name__, event.name,
                )
                payload = {}
                break
            payload = result

        if payload == original:
            new_events.append(event)
            continue

        changed = True
        new_events.append(
            Event(name=event.name, attributes=payload, timestamp=event.timestamp)
        )

    if not changed:
        return span

    obfuscated = copy.copy(span)
    obfuscated._events = tuple(new_events)
    setattr(obfuscated, _OBFUSCATED_MARKER, True)
    return obfuscated


def obfuscate_spans(
    spans: Sequence[ReadableSpan], obfuscators: Sequence[SpanObfuscator]
) -> Sequence[ReadableSpan]:
    """Apply *obfuscators* to every span, returning the spans to export."""
    if not obfuscators:
        return spans
    return [obfuscate_span(span, obfuscators) for span in spans]


class ObfuscatingSpanProcessor(SpanProcessor):
    """Span processor wrapper that obfuscates payloads on the way to ``on_end``.

    Monocle patches processors in place instead (see
    :func:`install_obfuscation_hooks`); this is the fallback for processors that
    cannot be patched, and an option for callers who prefer wrapping.
    """

    def __init__(self, processor: SpanProcessor, obfuscators: Sequence[SpanObfuscator]):
        self.processor = processor
        self.obfuscators = list(obfuscators)

    def on_start(self, span, parent_context=None) -> None:
        self.processor.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        self.processor.on_end(obfuscate_span(span, self.obfuscators))

    def shutdown(self) -> None:
        self.processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.processor.force_flush(timeout_millis)

    def __getattr__(self, name: str) -> Any:
        # Keep processor-specific helpers reachable (e.g. the trace-return processor's).
        return getattr(self.processor, name)


class ObfuscatingSpanExporter:
    """Exporter wrapper that obfuscates span payloads before delegating.

    For exporters wired up outside ``setup_monocle_telemetry``, which therefore
    miss the processor-level hook. Attributes other than the exporter interface
    are delegated to the wrapped exporter.
    """

    def __init__(self, base_exporter: Any, obfuscators: Sequence[SpanObfuscator]):
        self.base_exporter = base_exporter
        self.obfuscators = list(obfuscators)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.base_exporter.export(obfuscate_spans(spans, self.obfuscators))

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.base_exporter.force_flush(timeout_millis)

    def shutdown(self) -> None:
        return self.base_exporter.shutdown()

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not defined above, so exporter-specific helpers
        # (get_finished_spans, last_file_processed, ...) keep working.
        return getattr(self.base_exporter, name)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

_span_obfuscators: Optional[List[SpanObfuscator]] = None

#: Short names accepted by MONOCLE_SPAN_OBFUSCATORS, mapped to factories that take
#: an optional ``span_types`` keyword.
BUILTIN_OBFUSCATORS: Dict[str, Any] = {
    # The default. "regex" is an alias, naming the mechanism rather than the target.
    "credentials": lambda **kw: RegexSpanObfuscator(**kw),
    "regex": lambda **kw: RegexSpanObfuscator(**kw),
    "presidio": lambda **kw: PresidioSpanObfuscator(**kw),
}

#: What is enabled when nothing is configured. Obfuscation is on by default.
DEFAULT_OBFUSCATOR_NAME = "credentials"


def set_span_obfuscators(obfuscators: Optional[Sequence[SpanObfuscator]]) -> None:
    """Replace the registered obfuscators.

    ``None`` restores env-based config; ``[]`` disables obfuscation.
    """
    global _span_obfuscators
    if obfuscators is None:
        _span_obfuscators = None
        return
    for obfuscator in obfuscators:
        if not isinstance(obfuscator, SpanObfuscator):
            raise TypeError(
                f"Expected a SpanObfuscator instance, got {type(obfuscator).__name__}"
            )
    _span_obfuscators = list(obfuscators)


def register_span_obfuscator(obfuscator: SpanObfuscator) -> None:
    """Append an obfuscator to the registered list."""
    set_span_obfuscators(list(get_span_obfuscators()) + [obfuscator])


def get_span_obfuscators() -> List[SpanObfuscator]:
    """Return the registered obfuscators, loading env config on first use."""
    global _span_obfuscators
    if _span_obfuscators is None:
        _span_obfuscators = _load_obfuscators_from_env()
    return _span_obfuscators


def obfuscation_disabled_by_env() -> bool:
    """Return True if MONOCLE_DISABLE_SPAN_OBFUSCATION or an off-switch value of
    MONOCLE_SPAN_OBFUSCATORS turns obfuscation off."""
    if os.environ.get(DISABLE_OBFUSCATION_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return True
    return os.environ.get(SPAN_OBFUSCATORS_ENV, "").strip().lower() in OBFUSCATION_OFF_VALUES


def _load_obfuscators_from_env() -> List[SpanObfuscator]:
    """Build obfuscators from the environment.

    Obfuscation is on by default: with nothing configured this returns the
    :data:`DEFAULT_OBFUSCATOR_NAME` obfuscator, which redacts credentials only.
    """
    if obfuscation_disabled_by_env():
        logger.debug("Monocle span obfuscation is disabled by environment.")
        return []

    span_types_env = os.environ.get(OBFUSCATE_SPAN_TYPES_ENV, "").strip()
    span_types = (
        [t.strip() for t in span_types_env.split(",") if t.strip()]
        if span_types_env
        else None
    )

    configured = os.environ.get(SPAN_OBFUSCATORS_ENV, "").strip()
    entries = (
        [e.strip() for e in configured.split(",") if e.strip()]
        if configured
        else [DEFAULT_OBFUSCATOR_NAME]
    )

    obfuscators: List[SpanObfuscator] = []
    for entry in entries:
        try:
            obfuscators.append(_instantiate_obfuscator(entry, span_types))
        except Exception as ex:
            logger.warning(
                "Unable to load Monocle span obfuscator '%s': %s. "
                "Spans will be exported without it.", entry, ex,
            )
    return obfuscators


def _instantiate_obfuscator(
    entry: str, span_types: Optional[Sequence[str]]
) -> SpanObfuscator:
    """Build one obfuscator from a built-in name or a ``module:ClassName`` path."""
    kwargs = {"span_types": span_types} if span_types else {}
    factory = BUILTIN_OBFUSCATORS.get(entry.lower())
    if factory is not None:
        return factory(**kwargs)
    return _import_obfuscator(entry, span_types)


def _import_obfuscator(
    target: str, span_types: Optional[Sequence[str]]
) -> SpanObfuscator:
    """Import ``module:ClassName`` (or ``module.ClassName``) and instantiate it."""
    if ":" in target:
        module_name, _, class_name = target.partition(":")
    elif "." in target:
        module_name, _, class_name = target.rpartition(".")
    else:
        raise ValueError(
            f"'{target}' is not a known obfuscator name or a 'module:ClassName' path. "
            f"Known names: {sorted(BUILTIN_OBFUSCATORS)}"
        )
    obfuscator_class = getattr(import_module(module_name), class_name)
    if not (
        isinstance(obfuscator_class, type)
        and issubclass(obfuscator_class, SpanObfuscator)
    ):
        raise TypeError(f"{target} is not a SpanObfuscator subclass")
    if span_types:
        return obfuscator_class(span_types=span_types)
    return obfuscator_class()


def wrap_exporter_with_obfuscation(
    exporter: Any, obfuscators: Optional[Sequence[SpanObfuscator]] = None
) -> Any:
    """Wrap *exporter* so it obfuscates before exporting, for exporters wired up
    outside ``setup_monocle_telemetry``. Unchanged when obfuscation is off."""
    obfuscators = get_span_obfuscators() if obfuscators is None else obfuscators
    if not obfuscators or isinstance(exporter, ObfuscatingSpanExporter):
        return exporter
    return ObfuscatingSpanExporter(exporter, obfuscators)


def install_obfuscation_hook(
    processor: Any, obfuscators: Optional[Sequence[SpanObfuscator]] = None
) -> Any:
    """Patch ``processor.on_end`` in place so payloads are scrubbed before it runs.

    Patching rather than wrapping keeps the processor's type intact -- Monocle
    already patches ``on_start`` the same way -- so a caller that passed in a
    ``SimpleSpanProcessor`` gets one back. Idempotent per instance, and a no-op
    when obfuscation is off.
    """
    obfuscators = get_span_obfuscators() if obfuscators is None else obfuscators
    if not obfuscators or getattr(processor, _OBFUSCATION_HOOK_MARKER, False):
        return processor
    if isinstance(processor, ObfuscatingSpanProcessor):
        return processor

    original_on_end = processor.on_end

    def on_end(span: ReadableSpan) -> None:
        original_on_end(obfuscate_span(span, obfuscators))

    try:
        processor.on_end = on_end
        setattr(processor, _OBFUSCATION_HOOK_MARKER, True)
    except AttributeError:
        # A slotted or otherwise immutable processor can't be patched; wrap it.
        return ObfuscatingSpanProcessor(processor, obfuscators)
    return processor


def install_obfuscation_hooks(
    processors: Sequence[Any], obfuscators: Optional[Sequence[SpanObfuscator]] = None
) -> List[Any]:
    """Install the obfuscation hook on each span processor."""
    return [install_obfuscation_hook(p, obfuscators) for p in processors]
