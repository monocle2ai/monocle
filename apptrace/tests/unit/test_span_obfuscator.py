"""Unit tests for sensitive-data obfuscation of span exports."""

import json
import os
import re
from unittest.mock import patch

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext, TraceFlags

from monocle_apptrace.exporters.base_exporter import serialize_span
from monocle_apptrace.exporters.span_obfuscator import (
    DEFAULT_PATTERNS,
    ObfuscatingSpanExporter,
    ObfuscatingSpanProcessor,
    RegexSpanObfuscator,
    SpanObfuscator,
    TextSpanObfuscator,
    get_span_obfuscators,
    install_obfuscation_hook,
    install_obfuscation_hooks,
    obfuscate_span,
    obfuscate_spans,
    obfuscation_disabled_by_env,
    register_span_obfuscator,
    set_span_obfuscators,
    wrap_exporter_with_obfuscation,
)

# Real-shaped but fake credentials, so the patterns match without exposing anything.
MOCK_OPENAI_KEY = "sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
MOCK_ANTHROPIC_KEY = "sk-ant-api03-Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0"
MOCK_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
MOCK_GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnopqrstuvwxyz"
MOCK_PASSWORD = "hunter2-not-a-real-password"

OBFUSCATION_ENV_VARS = (
    "MONOCLE_SPAN_OBFUSCATORS",
    "MONOCLE_OBFUSCATE_SPAN_TYPES",
    "MONOCLE_DISABLE_SPAN_OBFUSCATION",
)


def make_span(span_type="inference", input_payload="hello", output_payload="world",
              extra_events=()):
    """Build a ReadableSpan with data.input, data.output and metadata events."""
    return ReadableSpan(
        name="test.span",
        context=SpanContext(trace_id=0x1234, span_id=0x5678, is_remote=False,
                            trace_flags=TraceFlags(1)),
        attributes={"span.type": span_type, "monocle_apptrace.version": "1.0.0"},
        events=[
            Event("data.input", {"input": input_payload}, timestamp=1),
            Event("data.output", {"response": output_payload}, timestamp=2),
            Event("metadata", {"completion_tokens": 10}, timestamp=3),
            *extra_events,
        ],
        resource=Resource.create({}),
    )


def event_attrs(span, name):
    """Return the attributes of the named event on *span*."""
    return next(event for event in span.events if event.name == name).attributes


def scrub(text, obfuscator=None):
    """Run one string through an obfuscator, bypassing the span plumbing."""
    return (obfuscator or RegexSpanObfuscator()).obfuscate_text(
        text, "input", "data.input", make_span()
    )


def clear_obfuscation_env():
    """Drop every obfuscation env var so the built-in defaults apply."""
    for key in OBFUSCATION_ENV_VARS:
        os.environ.pop(key, None)


class RecordingExporter:
    """Minimal SpanExporter that records what it was handed."""

    def __init__(self):
        self.spans = []
        self.custom_attribute = "delegated"

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        pass


class ExportingProcessor:
    """Minimal processor that forwards each ended span straight to an exporter."""

    def __init__(self, exporter):
        self.exporter = exporter

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        self.exporter.export([span])


class UpperObfuscator(TextSpanObfuscator):
    """Uppercases every string, so changes are obvious in assertions."""

    def obfuscate_text(self, text, key, event_name, span):
        return text.upper()


@pytest.fixture(autouse=True)
def reset_registry():
    """Keep the module-level obfuscator registry from leaking between tests."""
    set_span_obfuscators(None)
    yield
    set_span_obfuscators(None)


class TestObfuscateSpan:
    """Core mechanics: what gets rewritten, and what is left untouched."""

    def test_obfuscates_data_input_and_output_only(self):
        result = obfuscate_span(make_span(input_payload="in", output_payload="out"),
                                [UpperObfuscator()])

        assert event_attrs(result, "data.input")["input"] == "IN"
        assert event_attrs(result, "data.output")["response"] == "OUT"
        assert event_attrs(result, "metadata") == {"completion_tokens": 10}

    def test_does_not_mutate_the_original_span(self):
        span = make_span(input_payload="secret")
        obfuscate_span(span, [UpperObfuscator()])
        assert event_attrs(span, "data.input")["input"] == "secret"

    def test_preserves_span_identity_and_event_metadata(self):
        span = make_span()
        result = obfuscate_span(span, [UpperObfuscator()])

        assert (result.name, result.context, result.attributes) == (
            span.name, span.context, span.attributes)
        assert [(e.name, e.timestamp) for e in result.events] == [
            (e.name, e.timestamp) for e in span.events]

    @pytest.mark.parametrize("obfuscators,span_type", [
        ([], "inference"),                                     # nothing configured
        ([UpperObfuscator(span_types=["inference"])], "retrieval"),  # type mismatch
    ])
    def test_returns_the_same_object_when_nothing_applies(self, obfuscators, span_type):
        span = make_span(span_type)
        assert obfuscate_span(span, obfuscators) is span

    def test_unchanged_payload_returns_the_same_object(self):
        span = make_span(input_payload="HELLO", output_payload="WORLD")
        assert obfuscate_span(span, [UpperObfuscator()]) is span

    def test_is_idempotent(self):
        class Appender(TextSpanObfuscator):
            def obfuscate_text(self, text, key, event_name, span):
                return text + "!"

        once = obfuscate_span(make_span(input_payload="abc"), [Appender()])
        twice = obfuscate_span(once, [Appender()])

        assert twice is once
        assert event_attrs(twice, "data.input")["input"] == "abc!"

    def test_event_names_can_be_overridden(self):
        span = make_span(extra_events=[Event("data.internal", {"note": "x"}, timestamp=4)])
        result = obfuscate_span(span, [UpperObfuscator(event_names=["data.internal"])])

        assert event_attrs(result, "data.internal")["note"] == "X"
        assert event_attrs(result, "data.input")["input"] == "hello"

    def test_obfuscators_chain_in_order(self):
        class Prefix(TextSpanObfuscator):
            def __init__(self, prefix):
                super().__init__()
                self.prefix = prefix

            def obfuscate_text(self, text, key, event_name, span):
                return self.prefix + text

        result = obfuscate_span(make_span(input_payload="x"), [Prefix("a"), Prefix("b")])
        assert event_attrs(result, "data.input")["input"] == "bax"

    def test_obfuscate_spans_handles_a_mixed_batch(self):
        spans = [make_span("inference", "a"), make_span("retrieval", "b")]
        results = obfuscate_spans(spans, [UpperObfuscator(span_types=["inference"])])

        assert event_attrs(results[0], "data.input")["input"] == "A"
        assert results[1] is spans[1]

    def test_obfuscated_span_serializes_with_the_redacted_payload(self):
        result = obfuscate_span(make_span(input_payload="secret"), [UpperObfuscator()])
        assert "secret" not in json.dumps(serialize_span(result))

    def test_walks_strings_nested_in_lists_tuples_and_dicts(self):
        payload = ["a", ("b",), {"k": "c"}, 5, None, True]
        result = obfuscate_span(make_span(input_payload=payload), [UpperObfuscator()])

        assert event_attrs(result, "data.input")["input"] == [
            "A", ("B",), {"k": "C"}, 5, None, True]

    def test_payload_can_be_rewritten_wholesale(self):
        class Dropper(SpanObfuscator):
            def obfuscate(self, payload, event_name, span):
                return {}

        result = obfuscate_span(make_span(), [Dropper()])
        assert event_attrs(result, "data.input") == {}
        assert event_attrs(result, "metadata") == {"completion_tokens": 10}

    def test_the_payload_handed_to_an_obfuscator_is_a_copy(self):
        class Mutator(SpanObfuscator):
            def obfuscate(self, payload, event_name, span):
                payload["injected"] = "x"
                return payload

        span = make_span()
        obfuscate_span(span, [Mutator()])
        assert "injected" not in event_attrs(span, "data.input")

    @pytest.mark.parametrize("attrs", [
        {"events": ()},                                   # no events to scrub
        {"attributes": {"span.type": "inference"}},       # no events attribute at all
        {},                                               # neither
    ])
    def test_span_like_objects_without_events_pass_through(self, attrs):
        """Partial span objects reach exporters; obfuscation must not break the export."""
        span = type("PartialSpan", (), attrs)()
        assert obfuscate_span(span, [RegexSpanObfuscator()]) is span

    @pytest.mark.parametrize("broken", [
        lambda payload, event_name, span: 1 / 0,       # raises
        lambda payload, event_name, span: "not a dict",  # wrong return type
    ])
    def test_a_broken_obfuscator_drops_the_payload_instead_of_leaking_it(self, broken):
        class Broken(SpanObfuscator):
            obfuscate = staticmethod(broken)

        result = obfuscate_span(make_span(input_payload="secret"), [Broken()])
        assert event_attrs(result, "data.input") == {}
        assert event_attrs(result, "data.output") == {}


class TestCredentialRedaction:
    """Credentials are what the default pattern set exists to catch."""

    @pytest.mark.parametrize("text,expected", [
        (f"my key is {MOCK_OPENAI_KEY}", "<API_KEY>"),
        (f"anthropic key {MOCK_ANTHROPIC_KEY}", "<API_KEY>"),
        (f"{MOCK_AWS_KEY} is the key", "<AWS_ACCESS_KEY>"),
        ("AIzaSyA12345678901234567890123456789012", "<API_KEY>"),
        (MOCK_GITHUB_TOKEN, "<API_KEY>"),
        ("xoxb-1234567890-abcdefg", "<API_KEY>"),
        ("Authorization: Bearer abcdef1234567890", "Bearer <TOKEN>"),
        ("eyJhbGciOi.eyJzdWIiOi.SflKxwRJSM", "<JWT>"),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAK\n-----END RSA PRIVATE KEY-----",
         "<PRIVATE_KEY>"),
        ('{"api_key": "supersecretvalue"}', "<REDACTED>"),
        ('{"x-api-key": "supersecretvalue"}', "<REDACTED>"),
        (f"password={MOCK_PASSWORD}", "<REDACTED>"),
        ("client_secret: abc123def456", "<REDACTED>"),
        ("refresh_token=abc123def456", "<REDACTED>"),
    ])
    def test_redacts_credential_shapes(self, text, expected):
        assert expected in scrub(text)

    @pytest.mark.parametrize("secret,template", [
        (MOCK_OPENAI_KEY, "a {} b"),
        (MOCK_AWS_KEY, "a {} b"),
        (MOCK_GITHUB_TOKEN, "a {} b"),
        (MOCK_PASSWORD, "prefix password={} suffix"),
        ("letmein123", 'body {{"client_secret": "{}"}} end'),
    ])
    def test_the_secret_value_is_gone(self, secret, template):
        assert secret not in scrub(template.format(secret))

    def test_redacted_json_payload_stays_parseable(self):
        result = scrub(f'{{"model": "gpt-4", "api_key": "{MOCK_OPENAI_KEY}"}}')

        assert json.loads(result)["model"] == "gpt-4"
        assert MOCK_OPENAI_KEY not in result

    @pytest.mark.parametrize("text", [
        "What is the capital of France?",
        "The order total was 1234567890123456789 units",
        "Version 1.2.3 released on 2024-01-15",
        "The token count was 1500 and the prompt used 300",
        "Summarize the secret garden novel",           # 'secret' with no assignment
        "Contact sales@example.com for pricing",       # PII is not a credential
        "Her phone is 555-123-4567 at 10.0.0.1",
    ])
    def test_leaves_ordinary_text_alone(self, text):
        assert scrub(text) == text

    def test_end_to_end_on_a_span(self):
        span = make_span(
            input_payload=[f'{{"content": "key {MOCK_GITHUB_TOKEN}"}}'],
            output_payload=f"use key {MOCK_OPENAI_KEY}",
        )
        result = obfuscate_span(span, [RegexSpanObfuscator()])
        exported = json.dumps(serialize_span(result))

        assert MOCK_GITHUB_TOKEN not in exported and MOCK_OPENAI_KEY not in exported
        assert "<API_KEY>" in event_attrs(result, "data.output")["response"]

    def test_scoped_to_span_types(self):
        obfuscator = RegexSpanObfuscator(span_types=["inference", "inference.*"])
        inference = obfuscate_span(make_span("inference", MOCK_OPENAI_KEY), [obfuscator])
        retrieval = obfuscate_span(make_span("retrieval", MOCK_OPENAI_KEY), [obfuscator])

        assert "<API_KEY>" in event_attrs(inference, "data.input")["input"]
        assert event_attrs(retrieval, "data.input")["input"] == MOCK_OPENAI_KEY


class TestPatternSelection:
    def test_all_patterns_are_on_by_default(self):
        assert RegexSpanObfuscator().pattern_names == tuple(DEFAULT_PATTERNS)

    def test_patterns_can_be_narrowed(self):
        obfuscator = RegexSpanObfuscator(patterns=["github_token"])
        result = scrub(f"{MOCK_GITHUB_TOKEN} and {MOCK_OPENAI_KEY}", obfuscator)

        assert "<API_KEY>" in result
        assert MOCK_OPENAI_KEY in result

    def test_accepts_a_bare_name_and_collapses_duplicates(self):
        assert RegexSpanObfuscator(patterns="jwt").pattern_names == ("jwt",)
        assert RegexSpanObfuscator(patterns=["jwt", "jwt"]).pattern_names == ("jwt",)

    def test_order_follows_definition_order(self):
        # credential_assignment must run after the specific key shapes, or it
        # mangles them into an unrecognizable <REDACTED>.
        names = RegexSpanObfuscator(
            patterns=["credential_assignment", "openai_api_key"]).pattern_names
        assert names.index("openai_api_key") < names.index("credential_assignment")

    @pytest.mark.parametrize("name", ["not_a_pattern", "email", "credit_card", "us_ssn"])
    def test_unknown_pattern_name_raises(self, name):
        """Including names removed from the built-ins, so a typo can't pass silently."""
        with pytest.raises(ValueError, match="Unknown obfuscation pattern"):
            RegexSpanObfuscator(patterns=[name])

    def test_extra_patterns_are_applied(self):
        obfuscator = RegexSpanObfuscator(
            extra_patterns={"employee_id": (re.compile(r"\bEMP-\d{5}\b"), "<EMPLOYEE_ID>")})
        assert scrub("hired EMP-12345", obfuscator) == "hired <EMPLOYEE_ID>"

    def test_all_patterns_are_well_formed(self):
        for name, (regex, replacement) in DEFAULT_PATTERNS.items():
            assert hasattr(regex, "sub"), name
            assert callable(replacement) or isinstance(replacement, str), name


class TestPresidioSpanObfuscator:
    """Verifies the Presidio wiring without requiring presidio to be installed."""

    def test_uses_injected_engines(self):
        from monocle_apptrace.exporters.span_obfuscator import PresidioSpanObfuscator

        class FakeAnalyzer:
            def __init__(self):
                self.calls = []

            def analyze(self, text, entities, language, score_threshold):
                self.calls.append((entities, language, score_threshold))
                return ["hit"] if "bob@corp.com" in text else []

        class FakeAnonymizer:
            def anonymize(self, text, analyzer_results, operators):
                return type("Result", (), {
                    "text": text.replace("bob@corp.com", "<EMAIL_ADDRESS>")})()

        analyzer = FakeAnalyzer()
        obfuscator = PresidioSpanObfuscator(
            entities=["EMAIL_ADDRESS"], score_threshold=0.7,
            analyzer=analyzer, anonymizer=FakeAnonymizer())

        result = obfuscate_span(
            make_span(input_payload="write bob@corp.com", output_payload="ok"), [obfuscator])

        assert event_attrs(result, "data.input")["input"] == "write <EMAIL_ADDRESS>"
        assert analyzer.calls[0] == (["EMAIL_ADDRESS"], "en", 0.7)

    def test_blank_text_skips_analysis(self):
        from monocle_apptrace.exporters.span_obfuscator import PresidioSpanObfuscator

        class ExplodingAnalyzer:
            def analyze(self, **kwargs):
                raise AssertionError("should not analyze blank text")

        obfuscator = PresidioSpanObfuscator(analyzer=ExplodingAnalyzer(), anonymizer=object())
        assert obfuscator.obfuscate_text("   ", "input", "data.input", make_span()) == "   "

    def test_missing_dependency_raises_a_helpful_error(self):
        from monocle_apptrace.exporters import span_obfuscator

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("presidio"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="presidio-analyzer"):
                span_obfuscator.PresidioSpanObfuscator()


class TestHookAndWrapper:
    """Where obfuscation attaches: processors (patched) and exporters (wrapped)."""

    def test_hook_obfuscates_on_end(self):
        exporter = RecordingExporter()
        processor = install_obfuscation_hook(ExportingProcessor(exporter), [UpperObfuscator()])

        processor.on_end(make_span(input_payload="secret"))
        assert event_attrs(exporter.spans[0], "data.input")["input"] == "SECRET"

    def test_hook_preserves_the_processor_object_and_type(self):
        original = SimpleSpanProcessor(InMemorySpanExporter())
        hooked = install_obfuscation_hook(original, [UpperObfuscator()])

        assert hooked is original
        assert isinstance(hooked, SimpleSpanProcessor)

    def test_hooked_processor_still_exports(self):
        exporter = InMemorySpanExporter()
        processor = install_obfuscation_hook(
            SimpleSpanProcessor(exporter), [RegexSpanObfuscator()])

        processor.on_end(make_span(input_payload=f"key {MOCK_OPENAI_KEY}"))
        assert "<API_KEY>" in event_attrs(
            exporter.get_finished_spans()[0], "data.input")["input"]

    def test_hook_is_a_noop_when_obfuscation_is_off(self):
        set_span_obfuscators([])
        exporter = RecordingExporter()
        processor = ExportingProcessor(exporter)

        assert install_obfuscation_hook(processor) is processor
        processor.on_end(make_span(input_payload="secret"))
        assert event_attrs(exporter.spans[0], "data.input")["input"] == "secret"

    def test_hook_is_idempotent_per_processor(self):
        class Appender(TextSpanObfuscator):
            def obfuscate_text(self, text, key, event_name, span):
                return text + "!"

        exporter = RecordingExporter()
        processor = ExportingProcessor(exporter)
        for _ in range(2):
            install_obfuscation_hook(processor, [Appender()])

        processor.on_end(make_span(input_payload="abc"))
        assert event_attrs(exporter.spans[0], "data.input")["input"] == "abc!"

    def test_unpatchable_processor_falls_back_to_wrapping(self):
        class Slotted:
            __slots__ = ("exporter",)

            def __init__(self, exporter):
                self.exporter = exporter

            def on_end(self, span):
                self.exporter.export([span])

        exporter = RecordingExporter()
        result = install_obfuscation_hook(Slotted(exporter), [UpperObfuscator()])

        assert isinstance(result, ObfuscatingSpanProcessor)
        result.on_end(make_span(input_payload="secret"))
        assert event_attrs(exporter.spans[0], "data.input")["input"] == "SECRET"

    def test_install_hooks_covers_every_processor(self):
        exporters = [RecordingExporter(), RecordingExporter()]
        for processor in install_obfuscation_hooks(
            [ExportingProcessor(e) for e in exporters], [UpperObfuscator()]
        ):
            processor.on_end(make_span(input_payload="secret"))

        for exporter in exporters:
            assert event_attrs(exporter.spans[0], "data.input")["input"] == "SECRET"

    def test_exporter_wrapper_obfuscates_and_delegates(self):
        exporter = RecordingExporter()
        wrapped = ObfuscatingSpanExporter(exporter, [UpperObfuscator()])

        assert wrapped.export([make_span(input_payload="secret")]) == SpanExportResult.SUCCESS
        assert event_attrs(exporter.spans[0], "data.input")["input"] == "SECRET"
        assert wrapped.custom_attribute == "delegated"  # unknown attrs delegate

    def test_wrap_exporter_respects_the_registry(self):
        exporter = RecordingExporter()

        set_span_obfuscators([])
        assert wrap_exporter_with_obfuscation(exporter) is exporter

        set_span_obfuscators([UpperObfuscator()])
        wrapped = wrap_exporter_with_obfuscation(exporter)
        assert isinstance(wrapped, ObfuscatingSpanExporter)
        assert wrap_exporter_with_obfuscation(wrapped) is wrapped  # no double wrap

    def test_layering_a_hook_and_a_wrapper_obfuscates_once(self):
        class Appender(TextSpanObfuscator):
            def obfuscate_text(self, text, key, event_name, span):
                return text + "!"

        obfuscators = [Appender()]
        exporter = RecordingExporter()
        processor = install_obfuscation_hook(
            ExportingProcessor(ObfuscatingSpanExporter(exporter, obfuscators)), obfuscators)

        processor.on_end(make_span(input_payload="abc"))
        assert event_attrs(exporter.spans[0], "data.input")["input"] == "abc!"

    def test_base_exporter_hook_applies_registered_obfuscators(self, tmp_path):
        from monocle_apptrace.exporters.file_exporter import FileSpanExporter

        set_span_obfuscators([UpperObfuscator()])
        with patch.dict(os.environ, {"MONOCLE_TRACE_OUTPUT_PATH": str(tmp_path)}):
            exporter = FileSpanExporter()

        obfuscated = exporter.obfuscate_spans([make_span(input_payload="secret")])
        assert event_attrs(obfuscated[0], "data.input")["input"] == "SECRET"

    def test_every_monocle_exporter_obfuscates_on_export(self):
        """Guard: SpanExporterBase wraps export() for all subclasses, including new ones."""
        from monocle_apptrace.exporters.base_exporter import SpanExporterBase

        unhooked = [
            cls.__name__
            for cls in SpanExporterBase.__subclasses__()
            if "export" in cls.__dict__
            and not getattr(cls.__dict__["export"], "_monocle_obfuscates", False)
        ]
        assert not unhooked, f"exporters not routed through obfuscation: {unhooked}"

    def test_exporter_used_directly_scrubs_on_export(self):
        """A Monocle exporter used on its own, with no span processor, still scrubs."""
        from monocle_apptrace.exporters.base_exporter import MonocleInMemorySpanExporter

        set_span_obfuscators([RegexSpanObfuscator()])
        exporter = MonocleInMemorySpanExporter()
        exporter.export([make_span(input_payload=f"key {MOCK_OPENAI_KEY}")])

        finished = exporter.get_finished_spans()
        assert len(finished) == 1
        assert "<API_KEY>" in event_attrs(finished[0], "data.input")["input"]


class TestConfiguration:
    """Obfuscation must protect credentials with no configuration at all."""

    def test_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            clear_obfuscation_env()
            set_span_obfuscators(None)
            obfuscators = get_span_obfuscators()

            assert len(obfuscators) == 1
            assert isinstance(obfuscators[0], RegexSpanObfuscator)
            # Applies everywhere, since any span type can carry a leaked key.
            assert all(obfuscators[0].applies_to(make_span(t))
                       for t in ("inference", "retrieval", "agentic.invocation", "workflow"))

    @pytest.mark.parametrize("env", [
        {"MONOCLE_SPAN_OBFUSCATORS": "none"},
        {"MONOCLE_SPAN_OBFUSCATORS": "OFF"},
        {"MONOCLE_SPAN_OBFUSCATORS": "false"},
        {"MONOCLE_DISABLE_SPAN_OBFUSCATION": "true"},
        {"MONOCLE_DISABLE_SPAN_OBFUSCATION": "1"},
        # The disable flag wins over a configured obfuscator.
        {"MONOCLE_SPAN_OBFUSCATORS": "regex", "MONOCLE_DISABLE_SPAN_OBFUSCATION": "yes"},
    ])
    def test_disabled_by_env(self, env):
        with patch.dict(os.environ, env):
            set_span_obfuscators(None)
            assert obfuscation_disabled_by_env() is True
            assert get_span_obfuscators() == []

    @pytest.mark.parametrize("value", ["", "false-ish", "maybe"])
    def test_unrecognized_disable_value_leaves_it_enabled(self, value):
        """A typo in the disable flag must not silently drop protection."""
        with patch.dict(os.environ, {}, clear=False):
            clear_obfuscation_env()
            os.environ["MONOCLE_DISABLE_SPAN_OBFUSCATION"] = value
            set_span_obfuscators(None)
            assert obfuscation_disabled_by_env() is False
            assert get_span_obfuscators() != []

    @pytest.mark.parametrize("configured", [
        "regex",
        "credentials",
        "monocle_apptrace.exporters.span_obfuscator:RegexSpanObfuscator",
        "monocle_apptrace.exporters.span_obfuscator.RegexSpanObfuscator",
    ])
    def test_built_in_names_and_class_paths(self, configured):
        with patch.dict(os.environ, {"MONOCLE_SPAN_OBFUSCATORS": configured}):
            os.environ.pop("MONOCLE_DISABLE_SPAN_OBFUSCATION", None)
            set_span_obfuscators(None)
            assert isinstance(get_span_obfuscators()[0], RegexSpanObfuscator)

    def test_span_types_scoping_from_env(self):
        with patch.dict(os.environ, {
            "MONOCLE_SPAN_OBFUSCATORS": "regex",
            "MONOCLE_OBFUSCATE_SPAN_TYPES": "inference, inference.*",
        }):
            set_span_obfuscators(None)
            assert get_span_obfuscators()[0].span_types == ("inference", "inference.*")

    @pytest.mark.parametrize("configured,expected_warning", [
        ("nope,regex", "Unable to load Monocle span obfuscator 'nope'"),
        ("monocle_apptrace.exporters.span_filter:SpanFilter", "not a SpanObfuscator subclass"),
    ])
    def test_bad_entries_are_skipped_with_a_warning(self, configured, expected_warning, caplog):
        """A misconfigured obfuscator must not take down the traced app."""
        with patch.dict(os.environ, {"MONOCLE_SPAN_OBFUSCATORS": configured}):
            set_span_obfuscators(None)
            obfuscators = get_span_obfuscators()

        assert expected_warning in caplog.text
        assert all(isinstance(o, RegexSpanObfuscator) for o in obfuscators)

    def test_registry_set_and_register(self):
        first, second = UpperObfuscator(), RegexSpanObfuscator()

        set_span_obfuscators([first])
        assert get_span_obfuscators() == [first]

        register_span_obfuscator(second)
        assert get_span_obfuscators() == [first, second]

        set_span_obfuscators([])
        assert get_span_obfuscators() == []

    @pytest.mark.parametrize("register", [set_span_obfuscators, register_span_obfuscator])
    def test_registry_rejects_non_obfuscators(self, register):
        with pytest.raises(TypeError):
            register(["regex"] if register is set_span_obfuscators else "regex")

    def test_env_is_read_once_and_cached(self):
        with patch.dict(os.environ, {"MONOCLE_SPAN_OBFUSCATORS": "regex"}):
            set_span_obfuscators(None)
            assert get_span_obfuscators() is get_span_obfuscators()

    def test_exporter_factory_returns_concrete_exporter_types(self):
        """Obfuscation attaches to processors, so exporter types stay intact."""
        from monocle_apptrace.exporters.base_exporter import MonocleInMemorySpanExporter
        from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter

        with patch.dict(os.environ, {"MONOCLE_EXPORTER": "memory"}, clear=False):
            clear_obfuscation_env()
            set_span_obfuscators(None)
            exporters = get_monocle_exporter()

        assert isinstance(exporters[0], MonocleInMemorySpanExporter)


class TestSetupMonocleTelemetry:
    """End to end: a traced call must reach the exporter already scrubbed."""

    @pytest.fixture
    def traced(self):
        from monocle_apptrace.instrumentation.common.instrumentor import (
            get_monocle_instrumentor,
            setup_monocle_telemetry,
        )

        def uninstrument():
            instrumentor = get_monocle_instrumentor()
            if instrumentor is not None:
                try:
                    instrumentor.uninstrument()
                except Exception:
                    pass

        uninstrument()
        memory_exporter = InMemorySpanExporter()

        def setup(obfuscators=None):
            """``obfuscators=None`` exercises the default (env-driven) config."""
            setup_monocle_telemetry(
                workflow_name="span_obfuscator_test",
                span_processors=[SimpleSpanProcessor(memory_exporter)],
                union_with_default_methods=False,
                span_obfuscators=obfuscators,
            )
            return memory_exporter

        yield setup
        uninstrument()

    @staticmethod
    def events_for(exporter, span_name):
        for span in exporter.get_finished_spans():
            if span.name == span_name:
                return {e.name: dict(e.attributes) for e in span.events}
        raise AssertionError(
            f"no span named {span_name!r}; got "
            f"{[s.name for s in exporter.get_finished_spans()]}")

    def traced_call(self, traced, span_name, obfuscators=None, env=None):
        """Trace one call that leaks a key and a password, and return its events."""
        from monocle_apptrace.instrumentation.common.method_wrappers import (
            monocle_trace_method,
        )

        with patch.dict(os.environ, env or {}, clear=False):
            if env is None:
                clear_obfuscation_env()
            set_span_obfuscators(None)
            exporter = traced(obfuscators)

            @monocle_trace_method(span_name=span_name)
            def handler(payload):
                return f"retried with Authorization: Bearer {MOCK_ANTHROPIC_KEY}"

            handler({"user": "bob", "api_key": MOCK_OPENAI_KEY, "password": MOCK_PASSWORD})

        return self.events_for(exporter, span_name)

    def test_credentials_are_scrubbed_with_no_configuration(self, traced):
        events = self.traced_call(traced, "call_model")
        exported = json.dumps(events)

        for secret in (MOCK_OPENAI_KEY, MOCK_ANTHROPIC_KEY, MOCK_PASSWORD):
            assert secret not in exported, f"{secret} leaked into the exported span"
        assert "<API_KEY>" in events["data.input"]["input"]
        assert "<REDACTED>" in events["data.input"]["input"]
        assert "Bearer <TOKEN>" in events["data.output"]["response"]
        # Non-sensitive context survives, so the trace stays useful.
        assert "bob" in events["data.input"]["input"]

    def test_env_var_disables_obfuscation_end_to_end(self, traced):
        events = self.traced_call(traced, "unscrubbed",
                                  env={"MONOCLE_DISABLE_SPAN_OBFUSCATION": "true"})
        assert MOCK_OPENAI_KEY in events["data.input"]["input"]

    def test_empty_list_disables_obfuscation(self, traced):
        events = self.traced_call(traced, "plain", obfuscators=[])
        assert MOCK_OPENAI_KEY in events["data.input"]["input"]

    def test_explicit_obfuscators_override_the_environment(self, traced):
        events = self.traced_call(traced, "explicit", obfuscators=[RegexSpanObfuscator()],
                                  env={"MONOCLE_DISABLE_SPAN_OBFUSCATION": "true"})
        assert "<API_KEY>" in events["data.input"]["input"]

    def test_caller_supplied_processor_keeps_its_type(self, traced):
        """Hooking must not swap out the processor the caller handed to Monocle."""
        from monocle_apptrace.instrumentation.common.instrumentor import (
            get_monocle_span_processor,
        )

        traced([RegexSpanObfuscator()])
        processors = get_monocle_span_processor()._span_processors

        assert any(isinstance(p, SimpleSpanProcessor) for p in processors), (
            f"expected a SimpleSpanProcessor, got {[type(p).__name__ for p in processors]}")
