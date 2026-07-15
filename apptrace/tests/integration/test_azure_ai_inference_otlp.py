import os
import pytest

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# Configure Azure AI tracing
os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true"
os.environ["AZURE_SDK_TRACING_IMPLEMENTATION"] = "opentelemetry"

# Configure OLTP exporter for Monocle
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"

@pytest.mark.skipif(not GITHUB_TOKEN, reason="GITHUB_TOKEN not set")
def test_azure_ai_inference_otlp():
    # Import and setup Monocle
    from monocle_apptrace import setup_monocle_telemetry

    # Setup Monocle telemetry with OTLP exporter
    setup_monocle_telemetry(
        workflow_name="azure_ai_inference_with_monocle_otlp",
        monocle_exporters_list="otlp"
    )

    # Instrument Azure AI Inference
    from azure.ai.inference.tracing import AIInferenceInstrumentor
    AIInferenceInstrumentor().instrument()

    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import UserMessage
    from azure.ai.inference.models import TextContentItem
    from azure.core.credentials import AzureKeyCredential

    client = ChatCompletionsClient(
        endpoint = "https://models.inference.ai.azure.com",
        credential = AzureKeyCredential(GITHUB_TOKEN),
        api_version = "2024-08-01-preview",
    )

    response = client.complete(
    messages = [
        UserMessage(content = [
            TextContentItem(text = "hi"),
        ]),
    ],
    model = "gpt-4.1",
    tools = [],
    response_format = "text",
    temperature = 1,
    top_p = 1,
)

    assert response.choices[0].message.content
