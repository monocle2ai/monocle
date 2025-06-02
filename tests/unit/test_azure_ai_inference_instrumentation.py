import pytest
import time
from unittest.mock import Mock, patch
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from tests.common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="azure_ai_inference_test",
        span_processors=[BatchSpanProcessor(custom_exporter, max_queue_size=1, max_export_batch_size=1)],
        wrapper_methods=[]
    )


# Mock response classes for testing
class MockUsage:
    def __init__(self, completion_tokens=10, prompt_tokens=5, total_tokens=15):
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens


class MockMessage:
    def __init__(self, content="Test response"):
        self.content = content


class MockChoice:
    def __init__(self, message=None):
        self.message = message or MockMessage()


class MockChatCompletion:
    def __init__(self, choices=None, usage=None, model="test-model"):
        self.choices = choices or [MockChoice()]
        self.usage = usage or MockUsage()
        self.model = model


class MockEmbeddingData:
    def __init__(self, index=0, embedding=None):
        self.index = index
        self.embedding = embedding or [0.1, 0.2, 0.3, 0.4, 0.5]


class MockEmbeddingResponse:
    def __init__(self, data=None):
        self.data = data or [MockEmbeddingData()]


@pytest.mark.unit
def test_azure_ai_inference_chat_completion(setup):
    """Test Azure AI Inference chat completion instrumentation."""
    
    # Create a mock client
    client = ChatCompletionsClient(
        endpoint="https://test.models.ai.azure.com",
        credential=AzureKeyCredential("test-key")
    )
    
    # Mock the complete method to return our test response
    mock_response = MockChatCompletion()
    
    with patch.object(client, 'complete', return_value=mock_response) as mock_complete:
        # Make the chat completion call
        response = client.complete(
            messages=[
                SystemMessage("You are a helpful assistant."),
                UserMessage("What is the capital of France?"),
            ],
            model="gpt-4"
        )
        
        # Verify the mock was called
        mock_complete.assert_called_once()
        
        # Wait for spans to be processed
        time.sleep(2)
        
        # Get captured spans
        spans = custom_exporter.get_captured_spans()
        
        # Verify we have spans
        assert len(spans) > 0
        
        # Find the inference span
        inference_span = None
        for span in spans:
            span_attributes = span.attributes
            if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
                inference_span = span
                break
        
        # Verify inference span exists and has correct attributes
        assert inference_span is not None
        span_attributes = inference_span.attributes
        
        # Check span type and entity attributes
        assert span_attributes["span.type"] == "inference"
        assert "entity.1.type" in span_attributes
        assert span_attributes["entity.1.type"] == "inference.azure_ai_inference"
        assert "entity.1.inference_endpoint" in span_attributes
        assert "entity.2.name" in span_attributes
        assert "entity.2.type" in span_attributes
        
        # Check events
        assert len(inference_span.events) >= 2  # Should have input and output events
        
        # Verify input event
        input_event = next((e for e in inference_span.events if e.name == "data.input"), None)
        assert input_event is not None
        assert "input" in input_event.attributes
        
        # Verify output event  
        output_event = next((e for e in inference_span.events if e.name == "data.output"), None)
        assert output_event is not None
        assert "response" in output_event.attributes


@pytest.mark.unit
def test_azure_ai_inference_embeddings(setup):
    """Test Azure AI Inference embeddings instrumentation."""
    
    # Create a mock embeddings client
    client = EmbeddingsClient(
        endpoint="https://test.models.ai.azure.com",
        credential=AzureKeyCredential("test-key")
    )
    
    # Mock the embed method to return our test response
    mock_response = MockEmbeddingResponse()
    
    with patch.object(client, 'embed', return_value=mock_response) as mock_embed:
        # Make the embeddings call
        response = client.embed(
            input=["Hello world", "Test embedding"],
            model="text-embedding-ada-002"
        )
        
        # Verify the mock was called
        mock_embed.assert_called_once()
        
        # Wait for spans to be processed
        time.sleep(2)
        
        # Get captured spans
        spans = custom_exporter.get_captured_spans()
        
        # Verify we have spans
        assert len(spans) > 0
        
        # Find the retrieval span
        retrieval_span = None
        for span in spans:
            span_attributes = span.attributes
            if "span.type" in span_attributes and span_attributes["span.type"] == "retrieval":
                retrieval_span = span
                break
        
        # Verify retrieval span exists and has correct attributes
        assert retrieval_span is not None
        span_attributes = retrieval_span.attributes
        
        # Check span type and entity attributes
        assert span_attributes["span.type"] == "retrieval"
        assert "entity.1.type" in span_attributes
        assert span_attributes["entity.1.type"] == "inference.azure_ai_inference"
        assert "entity.1.inference_endpoint" in span_attributes
        assert "entity.2.name" in span_attributes
        assert "entity.2.type" in span_attributes
        
        # Check events
        assert len(retrieval_span.events) >= 2  # Should have input and output events
        
        # Verify input event
        input_event = next((e for e in retrieval_span.events if e.name == "data.input"), None)
        assert input_event is not None
        assert "input" in input_event.attributes
        
        # Verify output event
        output_event = next((e for e in retrieval_span.events if e.name == "data.output"), None)
        assert output_event is not None
        assert "response" in output_event.attributes


if __name__ == "__main__":
    pytest.main([__file__])
