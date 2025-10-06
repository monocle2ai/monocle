
import logging
import os
import time
import unittest
import warnings
from typing import List
from unittest.mock import MagicMock, patch

from common.mock_span_exporter import MockSpanExporter
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from openai.types.chat import ChatCompletion
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider

logger = logging.getLogger(__name__)

class TestHandler(unittest.TestCase):
    ragText = "sample_rag_text"
    instrumentor = None
    
    def setUp(self):
        """Set up test environment with clean state"""
        # Clean environment variables to ensure fresh state
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"
        os.environ["HTTP_API_KEY"] = "key1" 
        os.environ["OPENAI_API_KEY"] = "test-api-key-123"
        
        # Reset OpenTelemetry state - create fresh tracer provider
        tracer_provider = TracerProvider()
        set_tracer_provider(tracer_provider)
        
        # Suppress noisy instrumentation warnings
        logging.getLogger('monocle_apptrace.instrumentation.common.instrumentor').setLevel(logging.CRITICAL)
        logging.getLogger('haystack.components.builders.prompt_builder').setLevel(logging.CRITICAL)
        logging.getLogger('opentelemetry.attributes').setLevel(logging.CRITICAL)
        logging.getLogger('tests.common.http_span_exporter').setLevel(logging.CRITICAL)
        
        # Suppress pydantic warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._fields")
    
    def tearDown(self):
        """Clean up instrumentation state and prevent state leaks"""
        # First uninstrument if needed
        if self.instrumentor is not None:
            try:
                self.instrumentor.uninstrument()
                self.instrumentor = None
            except Exception as e:
                logger.warning(f"Uninstrument failed: {e}")
        
        # Clean up any global state that might affect other tests
        import gc
        gc.collect()
        
        # Reset warning filters
        warnings.resetwarnings()
        
    
    @patch('openai.resources.chat.completions.Completions.create')
    def test_haystack(self, mock_openai_create):
        # Set up environment
        api_key = os.getenv("OPENAI_API_KEY", "fake-api-key-for-testing")
        
        # Mock the OpenAI response - create a mock ChatCompletion object that passes isinstance checks
        mock_completion = MagicMock(spec=ChatCompletion)
        
        # Set up choices
        mock_choice = MagicMock()
        mock_choice.message.content = TestHandler.ragText
        mock_choice.message.role = 'assistant'
        mock_choice.finish_reason = 'stop'
        mock_completion.choices = [mock_choice]
        
        # Set up usage
        mock_usage = MagicMock()
        mock_usage.completion_tokens = 10
        mock_usage.prompt_tokens = 20
        mock_usage.total_tokens = 30
        mock_completion.usage = mock_usage
        
        # Set other attributes
        mock_completion.model = "gpt-4"
        mock_completion.id = "test-completion-id"
        
        mock_openai_create.return_value = mock_completion

        # Use MockSpanExporter instead of HttpSpanExporter to avoid connection errors
        mock_exporter = MockSpanExporter()
        span_processor = BatchSpanProcessor(mock_exporter)
        self.instrumentor = setup_monocle_telemetry(
            workflow_name="haystack_app_1",
            span_processors=[span_processor],
            wrapper_methods=[]
        )
        # Create a simple pipeline without retriever for now, just test the chat functionality
        prompt_builder = ChatPromptBuilder()
        llm = OpenAIChatGenerator(api_key=Secret.from_token(api_key), model="gpt-4")

        pipe = Pipeline()
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.add_component("llm", llm)
        pipe.connect("prompt_builder.prompt", "llm.messages")
        
        query = "OpenTelemetry"
        messages = [ChatMessage.from_user("Tell me a joke about {{query}}")]
        pipe.run(
            data={
                "prompt_builder": {
                    "template": messages,
                    "template_variables": {"query": query}
                }
            }
        )

        time.sleep(3)

        # Force flush the span processor to ensure spans are exported
        span_processor.force_flush()

        # Get the exported spans from the mock exporter
        exported_batches = mock_exporter.get_exported_spans()
        assert len(exported_batches) > 0, "No spans were exported to the mock exporter"
        
        # Get the latest batch of spans
        dataJson = exported_batches[-1]
        assert 'batch' in dataJson, f"No 'batch' key in exported data: {dataJson}"
        assert len(dataJson['batch']) > 0, "No spans in the exported batch"

        root_attributes = [x for x in dataJson["batch"] if x["parent_id"] == "None"][0]["attributes"]

        # Check that we have at least the expected number of spans (could be more due to additional instrumentation)
        actual_span_count = len(dataJson['batch'])
        assert actual_span_count >= 3, f"Expected at least 3 spans, got {actual_span_count}"
        # Check that we have the expected span names - use partial matching for flexibility
        span_names: List[str] = [span.get("name", "unknown") for span in dataJson['batch']]
        expected_patterns = [
            "openai",  # Could be various OpenAI spans
            "OpenAIChatGenerator",  # Haystack OpenAI generator
            "Pipeline",  # Haystack pipeline
            "workflow"  # Main workflow span
        ]
        for pattern in expected_patterns:
            found = any(pattern in span_name for span_name in span_names)
            assert found, f"Expected pattern '{pattern}' not found in any of these spans: {span_names}"

        type_found = False
        model_name_found = False
        
        # Check workflow attributes that actually exist
        assert root_attributes["workflow.name"] == "haystack_app_1"
        assert root_attributes["span.type"] == "workflow"

        for span in dataJson["batch"]:
            if span.get("name") == "workflow" and "entity.1.type" in span.get("attributes", {}):
                assert span["attributes"]["entity.1.type"] == "workflow.haystack"
                type_found = True
            if span.get("name") == "haystack.components.generators.chat.openai.OpenAIChatGenerator.run" and "entity.2.name" in span.get("attributes", {}):
                assert span["attributes"]["entity.2.name"] == "gpt-4"
                model_name_found = True

        assert type_found
        assert model_name_found

if __name__ == '__main__':
    unittest.main()
