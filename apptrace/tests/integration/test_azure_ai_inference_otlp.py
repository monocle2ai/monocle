import os

# Configure Azure AI tracing
os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true"
os.environ["AZURE_SDK_TRACING_IMPLEMENTATION"] = "opentelemetry"

# Configure OLTP exporter for Monocle
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"

# Import and setup Monocle
from monocle_apptrace import setup_monocle_telemetry

# Setup Monocle telemetry with OTLP exporter
setup_monocle_telemetry(
    workflow_name="azure_ai_inference_with_monocle_otlp",
    monocle_exporters_list="otlp"
)

print("Monocle configured with OTLP exporter")
print(f"Traces will be sent to: {os.environ['OTEL_EXPORTER_OTLP_ENDPOINT']}")

github_token = os.environ["GITHUB_TOKEN"]

# Instrument Azure AI Inference
from azure.ai.inference.tracing import AIInferenceInstrumentor
AIInferenceInstrumentor().instrument()
### Set up for Monocle with OTLP Exporter ###

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.ai.inference.models import TextContentItem
from azure.core.credentials import AzureKeyCredential

client = ChatCompletionsClient(
    endpoint = "https://models.inference.ai.azure.com",
    credential = AzureKeyCredential(github_token),
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

print(response.choices[0].message.content)
