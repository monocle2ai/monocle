"""
Test cases to verify the async context detachment fix.

This test validates:
1. The monkey-patch for safe detach works correctly
2. Async functions work without context errors
3. Existing functionality is not broken
"""
import asyncio
import logging
import os
import unittest
import warnings
from opentelemetry import trace
from opentelemetry.context import attach, detach, set_value, get_value, contextvars_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import set_tracer_provider
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

logger = logging.getLogger(__name__)


class TestAsyncContextFix(unittest.IsolatedAsyncioTestCase):
    """Test suite for async context detachment fix."""
    
    instrumentor = None
    tracer = None
    
    @classmethod
    def setUpClass(cls):
        """Set up monocle telemetry once for all tests to apply the monkey-patch."""
        # Set up clean environment
        os.environ["HTTP_API_KEY"] = "test-key-123"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"
        
        # Suppress noisy warnings
        logging.getLogger('opentelemetry.trace').setLevel(logging.ERROR)
        logging.getLogger('google.adk.models.registry').setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._fields")
        
        # Set up TracerProvider before setup_monocle_telemetry
        tracer_provider = TracerProvider()
        set_tracer_provider(tracer_provider)
        
        # Set up monocle telemetry
        cls.instrumentor = setup_monocle_telemetry(
            workflow_name="test_async_context_fix",
            span_processors=[],
            wrapper_methods=[]
        )
        cls.tracer = trace.get_tracer(__name__)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up instrumentation after all tests."""
        if cls.instrumentor and cls.instrumentor.is_instrumented_by_opentelemetry:
            cls.instrumentor.uninstrument()
        cls.instrumentor = None
        cls.tracer = None
    
    def test_safe_detach_method_exists(self):
        """Verify the monkey-patch was applied to OpenTelemetry's detach."""
        self.assertTrue(hasattr(contextvars_context.ContextVarsRuntimeContext, '_original_detach'))
        self.assertNotEqual(
            contextvars_context.ContextVarsRuntimeContext.detach,
            contextvars_context.ContextVarsRuntimeContext._original_detach
        )
    
    async def test_async_function_with_context(self):
        """Test async function with context attach/detach."""
        async def async_operation():
            token = attach(set_value("async_key", "async_value"))
            await asyncio.sleep(0.01)
            value = get_value("async_key")
            detach(token)
            return value
        
        result = await async_operation()
        self.assertEqual(result, "async_value")
    
    async def test_concurrent_async_operations(self):
        """Test concurrent async operations with separate contexts."""
        async def async_task(task_id):
            token = attach(set_value(f"task_{task_id}", f"value_{task_id}"))
            await asyncio.sleep(0.01)
            value = get_value(f"task_{task_id}")
            detach(token)
            return value
        
        results = await asyncio.gather(async_task(1), async_task(2), async_task(3))
        self.assertEqual(results, ["value_1", "value_2", "value_3"])
    
    async def test_span_creation_in_async(self):
        """Test that span creation works correctly in async context."""
        async def create_span():
            with self.tracer.start_as_current_span("test_span") as span:
                self.assertTrue(span.is_recording())
                await asyncio.sleep(0.01)
                span.set_attribute("test_attr", "test_value")
                return "span_created", span
        
        result, span = await create_span()
        self.assertEqual(result, "span_created")
        self.assertEqual(span.attributes["test_attr"], "test_value")
    
    def test_sync_context_operations_still_work(self):
        """Verify synchronous context operations weren't broken."""
        def sync_operation():
            token = attach(set_value("sync_key", "sync_value"))
            value = get_value("sync_key")
            detach(token)
            return value
        
        result = sync_operation()
        self.assertEqual(result, "sync_value")


if __name__ == "__main__":
    unittest.main()
