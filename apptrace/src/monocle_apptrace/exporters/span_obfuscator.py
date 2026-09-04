"""
Scrubs sensitive data from span ``data.input`` / ``data.output`` payloads before
the span reaches any exporter.

On by default, redacting credentials -- API keys, passwords, tokens and private
keys -- and replacing inline media with its type and size. For PII (names,
addresses, national IDs) add the Presidio obfuscator, which needs the optional
``obfuscation`` extra.

    MONOCLE_DISABLE_SPAN_OBFUSCATION=true         # off
    MONOCLE_SPAN_OBFUSCATORS=credentials,presidio # add PII detection

An obfuscator is any subclass of :class:`SpanObfuscator`; subclass
:class:`TextSpanObfuscator` to rewrite strings, :class:`SpanObfuscator` for full
control over the payload dict.

See docs/monocle_span_obfuscation.md for configuration and examples.
"""

import copy
import json
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
OBFUSCATE_ENV_VALUES_ENV = "MONOCLE_OBFUSCATE_ENV_VALUES"
EXTRA_PATTERNS_ENV = "MONOCLE_OBFUSCATE_EXTRA_PATTERNS"

#: Shortest env var value redacted literally. A var set to "1" or "true" would
#: otherwise blank that text out of every span.
MIN_ENV_VALUE_LENGTH = 8

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


# Sentence punctuation an unquoted value runs into, since the value pattern stops
# only at whitespace and , ; } quotes.
_TRAILING_PUNCTUATION = ".:!?"


def _redact_credential_assignment(match: "re.Match") -> str:
    """Redact the value of an ``api_key: ...`` style assignment.

    Keeps the quotes around a quoted value so redacting a secret inside a JSON
    payload leaves it parseable, keeps sentence punctuation that an unquoted
    value ran into, and leaves the match alone when a more specific pattern
    already replaced the value -- so ``Authorization: Bearer <TOKEN>`` keeps its
    precise marker instead of degrading to ``Authorization: <REDACTED>``.
    """
    prefix, value = match.group(1), match.group(2)

    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        quote = value[0]
        if _is_already_handled(value.strip("\"'")):
            return match.group(0)
        return f"{prefix}{quote}<REDACTED>{quote}"

    value, trailing = _split_trailing_punctuation(value)
    if not value or _is_already_handled(value):
        return match.group(0)
    return f"{prefix}<REDACTED>{trailing}"


def _is_already_handled(value: str) -> bool:
    """True if *value* is a redaction placeholder or a bare auth scheme keyword."""
    return bool(_PLACEHOLDER.match(value)) or value.lower() in _AUTH_SCHEMES


def _split_trailing_punctuation(value: str) -> tuple:
    """Split trailing sentence punctuation off *value*."""
    stripped = value.rstrip(_TRAILING_PUNCTUATION)
    return stripped, value[len(stripped):]


#: Read by both ``credential_key`` and ``credential_assignment``, so the two
#: cannot drift apart.
_CREDENTIAL_KEY_NAMES = (
    r"api[_-]?key|apikey|secret|password|passwd|pwd|"
    r"access[_-]?token|auth[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"bearer[_-]?token|session[_-]?token|sas[_-]?token|"
    r"client[_-]?secret|private[_-]?key|credentials?|authorization"
)

CREDENTIAL_KEY_PATTERN = "credential_key"

#: Shortest base64 run treated as a blob. Prose never runs this far without a
#: space, and every credential shape is shorter and matched first.
BASE64_BLOB_MIN_LENGTH = 512


def _format_size(num_bytes: int) -> str:
    """Render a byte count the way a human reads it."""
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def _decoded_size(encoded: str) -> int:
    """Decoded byte count of a base64 string, without decoding it."""
    return len(encoded.rstrip("=")) * 3 // 4


def _redact_data_url(match: "re.Match") -> str:
    """Replace a ``data:`` URL with its media type and size."""
    media_type = (match.group(1) or "application/octet-stream").lower()
    label = "IMAGE" if media_type.startswith("image/") else "MEDIA"
    return f"<{label}:{media_type},{_format_size(_decoded_size(match.group(2)))}>"


def _redact_base64_blob(match: "re.Match") -> str:
    """Replace a bare base64 blob with its size.

    No media type: it sits in a sibling field this leaves alone, so Anthropic's
    ``media_type`` and Gemini's ``mime_type`` stay readable next to the marker.
    """
    return f"<BASE64:{_format_size(_decoded_size(match.group(0)))}>"


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
    # Monocle exports to Okahu, so its key is the one most likely to reach a
    # span -- and, passed positionally, the one no other pattern can name.
    "okahu_api_key": (re.compile(r"\bokh_[A-Za-z0-9_-]{16,}\b"), "<API_KEY>"),
    # The inline image, audio clip or PDF a multimodal call sends, as OpenAI's
    # image_url/input_image blocks carry it.
    "data_url": (
        re.compile(r"data:([\w.+-]+/[\w.+-]+)?;base64,([A-Za-z0-9+/]+={0,2})", re.I),
        _redact_data_url,
    ),
    # The same content with no data: prefix, as Anthropic's source.data and
    # Gemini's inline_data.data send it.
    "base64_blob": (
        re.compile(
            rf"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{{{BASE64_BLOB_MIN_LENGTH},}}={{0,2}}"
            r"(?![A-Za-z0-9+/])"
        ),
        _redact_base64_blob,
    ),
    # Generic "api_key": "value" / password=value assignments, including the
    # quoted form found in JSON payloads and the "x-api-key: value" header form.
    "credential_assignment": (
        re.compile(
            rf"(?i)\b((?:{_CREDENTIAL_KEY_NAMES})"
            r"\b[\"']?\s*[:=]\s*)"
            r"(\"[^\"]+\"|'[^']+'|[^\s,;}\"']+)"
        ),
        _redact_credential_assignment,
    ),
    # Matched against the payload key, not the text -- see obfuscate_text. Last,
    # so it only sees a value no shape pattern recognized. A credential name has
    # to end the key: "openai_api_key" matches, "password_hint" does not.
    CREDENTIAL_KEY_PATTERN: (
        re.compile(rf"(?i)^(?:.*[._\-])?(?:{_CREDENTIAL_KEY_NAMES})$"),
        "<REDACTED>",
    ),
}

#: Patterns that replace an encoded blob with its media type and size.
MEDIA_PATTERNS = ("data_url", "base64_blob")

#: Everything else -- the credential shapes and the two key-driven patterns.
CREDENTIAL_PATTERNS = tuple(
    name for name in DEFAULT_PATTERNS if name not in MEDIA_PATTERNS
)

#: All built-in patterns are on by default.
DEFAULT_ENABLED_PATTERNS = tuple(DEFAULT_PATTERNS)


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

    Most patterns match the text. ``credential_key`` matches the payload key
    instead, so a bare secret with no recognizable shape is still redacted when
    the key names it -- the ``{"api_key": "hunter2"}`` a custom output processor
    can produce. :data:`MEDIA_PATTERNS` are not about secrets: they replace an
    inline base64 blob with ``<IMAGE:image/png,1.4MB>``.
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
        self.key_pattern: Optional[Any] = None
        self.key_replacement: str = ""
        self.patterns: List[Any] = []
        for name in self.pattern_names:
            regex, replacement = DEFAULT_PATTERNS[name]
            if name == CREDENTIAL_KEY_PATTERN:
                self.key_pattern, self.key_replacement = regex, replacement
            else:
                self.patterns.append((regex, replacement))
        if extra_patterns:
            self.patterns.extend(extra_patterns.values())

    def redacts_key(self, key: str) -> bool:
        """True if *key* names a credential, making its whole value the secret."""
        return bool(self.key_pattern and key and self.key_pattern.match(key))

    def obfuscate_text(
        self, text: str, key: str, event_name: str, span: ReadableSpan
    ) -> str:
        for regex, replacement in self.patterns:
            text = regex.sub(replacement, text)
        # The key names a credential, so the whole value is the secret. A value
        # a pattern above replaced outright keeps that marker -- <API_KEY> does
        # not degrade to <REDACTED>, and repeated passes stay idempotent. One it
        # only partly matched is redacted whole, since the key covers all of it.
        if self.redacts_key(key) and text.strip() and not _is_already_handled(
            text.strip("\"'")
        ):
            return self.key_replacement
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

#: Names that build a RegexSpanObfuscator, the one that accepts extra patterns.
REGEX_OBFUSCATOR_NAMES = ("credentials", "regex")

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


def _patterns_from_env_values() -> Dict[str, Any]:
    """Patterns redacting the literal values of the env vars named by
    ``MONOCLE_OBFUSCATE_ENV_VALUES``.

    An app already holds its own secrets, so this needs no regex and catches
    them wherever they land -- including a positional argument, where no name
    identifies them. Values are never logged.
    """
    names = [
        name.strip()
        for name in os.environ.get(OBFUSCATE_ENV_VALUES_ENV, "").split(",")
        if name.strip()
    ]
    patterns: Dict[str, Any] = {}
    for name in names:
        value = os.environ.get(name, "").strip()
        if not value:
            logger.warning(
                "%s names '%s', which is not set; skipping it.",
                OBFUSCATE_ENV_VALUES_ENV, name,
            )
            continue
        if len(value) < MIN_ENV_VALUE_LENGTH:
            logger.warning(
                "Value of '%s' is shorter than %d characters; skipping it rather "
                "than redacting that text out of every span.",
                name, MIN_ENV_VALUE_LENGTH,
            )
            continue
        patterns[f"env:{name}"] = (re.compile(re.escape(value)), "<REDACTED>")
    return patterns


def _patterns_from_env_json() -> Dict[str, Any]:
    """Patterns from ``MONOCLE_OBFUSCATE_EXTRA_PATTERNS``, a JSON object of
    ``{"name": "regex"}`` or ``{"name": {"pattern": ..., "replacement": ...}}``.

    JSON rather than a delimited list because regexes contain the commas and
    colons such a list would split on.
    """
    raw = os.environ.get(EXTRA_PATTERNS_ENV, "").strip()
    if not raw:
        return {}
    try:
        spec = json.loads(raw)
    except ValueError as ex:
        logger.warning("%s is not valid JSON: %s. Ignoring it.", EXTRA_PATTERNS_ENV, ex)
        return {}
    if not isinstance(spec, dict):
        logger.warning(
            "%s must be a JSON object of {name: regex}, got %s. Ignoring it.",
            EXTRA_PATTERNS_ENV, type(spec).__name__,
        )
        return {}

    patterns: Dict[str, Any] = {}
    for name, entry in spec.items():
        if isinstance(entry, str):
            pattern, replacement = entry, "<REDACTED>"
        elif isinstance(entry, dict):
            pattern = entry.get("pattern")
            replacement = entry.get("replacement", "<REDACTED>")
        else:
            pattern = None
        if not isinstance(pattern, str) or not isinstance(replacement, str):
            logger.warning(
                "%s entry '%s' must be a regex string or {\"pattern\": ..., "
                "\"replacement\": ...}; skipping it.", EXTRA_PATTERNS_ENV, name,
            )
            continue
        try:
            patterns[name] = (re.compile(pattern), replacement)
        except re.error as ex:
            logger.warning(
                "%s entry '%s' is not a valid regex: %s. Skipping it.",
                EXTRA_PATTERNS_ENV, name, ex,
            )
    return patterns


def extra_patterns_from_env() -> Dict[str, Any]:
    """Every ``{name: (regex, replacement)}`` configured through the environment."""
    return {**_patterns_from_env_values(), **_patterns_from_env_json()}


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

    extra_patterns = extra_patterns_from_env()

    obfuscators: List[SpanObfuscator] = []
    for entry in entries:
        try:
            obfuscators.append(
                _instantiate_obfuscator(entry, span_types, extra_patterns)
            )
        except Exception as ex:
            logger.warning(
                "Unable to load Monocle span obfuscator '%s': %s. "
                "Spans will be exported without it.", entry, ex,
            )

    # Env-configured patterns ride on the regex obfuscator. If none was selected
    # -- MONOCLE_SPAN_OBFUSCATORS=presidio, say -- add one carrying only those,
    # so configuring a pattern always means it is applied. It goes first: a
    # literal env value must be matched before another obfuscator rewrites the
    # text out from under it.
    if extra_patterns and not any(
        isinstance(obf, RegexSpanObfuscator) for obf in obfuscators
    ):
        obfuscators.insert(
            0,
            RegexSpanObfuscator(
                patterns=[], extra_patterns=extra_patterns, span_types=span_types
            ),
        )
    return obfuscators


def _instantiate_obfuscator(
    entry: str,
    span_types: Optional[Sequence[str]],
    extra_patterns: Optional[Dict[str, Any]] = None,
) -> SpanObfuscator:
    """Build one obfuscator from a built-in name or a ``module:ClassName`` path."""
    kwargs = {"span_types": span_types} if span_types else {}
    factory = BUILTIN_OBFUSCATORS.get(entry.lower())
    if factory is not None:
        if extra_patterns and entry.lower() in REGEX_OBFUSCATOR_NAMES:
            kwargs["extra_patterns"] = extra_patterns
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
