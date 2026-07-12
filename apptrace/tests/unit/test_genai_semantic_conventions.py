from types import SimpleNamespace
from unittest.mock import patch

from monocle_apptrace.instrumentation.common.genai_semantic_conventions import (
    configure_otel_genai_semconv,
    enrich_genai_attributes,
)


class FakeSpan:
    def __init__(self, name, attributes, events=None):
        self.name = name
        self.attributes = dict(attributes)
        self.events = events or []

    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_inference_span_gets_standard_genai_attributes():
    configure_otel_genai_semconv(True, ())
    span = FakeSpan(
        "openai.chat.completions.create",
        {
            "span.type": "inference",
            "entity.1.provider_name": "openrouter.ai",
            "entity.1.type": "inference.openai",
            "entity.2.name": "openai/gpt-4.1-mini",
            "entity.2.type": "model.llm.openai/gpt-4.1-mini",
            "workflow.name": "trip_concierge",
        },
        events=[
            SimpleNamespace(
                name="metadata",
                attributes={"prompt_tokens": 12, "completion_tokens": 7},
            )
        ],
    )

    enrich_genai_attributes(span)

    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.workflow.name"] == "trip_concierge"
    assert span.attributes["gen_ai.provider.name"] == "openrouter.ai"
    assert span.attributes["gen_ai.request.model"] == "openai/gpt-4.1-mini"
    assert span.attributes["gen_ai.usage.input_tokens"] == 12
    assert span.attributes["gen_ai.usage.output_tokens"] == 7


def test_tool_span_gets_execute_tool_operation_and_name():
    configure_otel_genai_semconv(True, ())
    span = FakeSpan(
        "weather_tool",
        {
            "span.type": "agentic.tool.invocation",
            "entity.1.name": "get_weather",
            "entity.1.type": "tool.function",
        },
    )

    enrich_genai_attributes(span)

    assert span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert span.attributes["gen_ai.tool.name"] == "get_weather"


def test_embedding_span_uses_embeddings_operation_and_model():
    configure_otel_genai_semconv(True)
    span = FakeSpan(
        "openai.resources.embeddings.Embeddings",
        {
            "span.type": "embedding.modelapi",
            "entity.23.name": "text-embedding-3-large",
            "entity.23.type": "model.embedding.text-embedding-3-large",
        },
    )

    enrich_genai_attributes(span)

    assert span.attributes["gen_ai.operation.name"] == "embeddings"
    assert span.attributes["gen_ai.request.model"] == "text-embedding-3-large"


def test_retrieval_span_uses_retrieval_operation():
    configure_otel_genai_semconv(True)
    span = FakeSpan("vectorstore.search", {"span.type": "retrieval"})

    enrich_genai_attributes(span)

    assert span.attributes["gen_ai.operation.name"] == "retrieval"


def test_existing_standard_attributes_are_not_replaced():
    configure_otel_genai_semconv(True, ())
    span = FakeSpan(
        "inference",
        {
            "span.type": "inference",
            "gen_ai.operation.name": "custom_operation",
        },
    )

    enrich_genai_attributes(span)

    assert span.attributes["gen_ai.operation.name"] == "custom_operation"


def test_non_ai_span_is_unchanged():
    configure_otel_genai_semconv(True, ())
    attributes = {"span.type": "http.send"}
    span = FakeSpan("http.send", attributes)

    enrich_genai_attributes(span)

    assert span.attributes == attributes


def test_auto_mode_follows_builtin_otlp_exporter():
    with patch.dict("os.environ", {}, clear=True):
        assert configure_otel_genai_semconv(None, ("okahu", "otlp")) is True
        assert configure_otel_genai_semconv(None, ("okahu",)) is False


def test_environment_can_override_auto_mode():
    with patch.dict("os.environ", {"MONOCLE_OTEL_GENAI_SEMCONV": "false"}, clear=True):
        assert configure_otel_genai_semconv(None, ("otlp",)) is False

    with patch.dict("os.environ", {"MONOCLE_OTEL_GENAI_SEMCONV": "true"}, clear=True):
        assert configure_otel_genai_semconv(None, ("file",)) is True


def test_disabled_semantic_conventions_leave_ai_span_unchanged():
    configure_otel_genai_semconv(False, ("otlp",))
    attributes = {"span.type": "inference"}
    span = FakeSpan("inference", attributes)

    enrich_genai_attributes(span)

    assert span.attributes == attributes
