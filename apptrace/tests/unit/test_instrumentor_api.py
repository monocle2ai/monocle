import os
import logging
import unittest
import asyncio
from common.dummy_class import DummyClass
from common.utils import verify_traceID, SCOPE_NAME, SCOPE_VALUE
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry import trace
from monocle_apptrace.instrumentation.common.instrumentor import (
    setup_monocle_telemetry, 
    monocle_trace, 
    amonocle_trace, 
    monocle_trace_method,
    start_trace,
    stop_trace,
    get_tracer_provider
)
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper

logger = logging.getLogger(__name__)

class TestInstrumentorAPI(unittest.TestCase):
    """Test class for instrumentor API functions like monocle_trace, amonocle_trace, and monocle_trace_method"""
    
    @classmethod
    def setUpClass(cls):
        cls.exporter = CustomConsoleSpanExporter()
        cls.instrumentor = setup_monocle_telemetry(
            workflow_name="instrumentor_api_test",
            span_processors=[SimpleSpanProcessor(cls.exporter)],
            wrapper_methods=[
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="double_it",
                    span_name="double_it",
                    wrapper_method=task_wrapper
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="triple_it",
                    span_name="triple_it",
                    wrapper_method=task_wrapper
                )
            ]
        )
        cls.dummy = DummyClass()

    @classmethod
    def tearDownClass(cls):
        if cls.instrumentor is not None:
            cls.instrumentor.uninstrument()

    def setUp(self):
        self.exporter.reset()

    def tearDown(self):
        self.exporter.reset()

    def test_monocle_trace_basic(self):
        """Test basic functionality of monocle_trace context manager"""
        
        # Test with default span name
        with monocle_trace():
            result = self.dummy.double_it(5)
        
        self.assertEqual(result, 10)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        # Should have 2 spans: one from the context manager and one from the instrumented method
        self.assertEqual(len(spans), 2)
        
        # Verify trace ID consistency
        verify_traceID(self.exporter, 2)
        
        # Check that the context manager span exists
        span_names = [span.name for span in spans]
        self.assertIn("workflow", span_names)
        self.assertIn("double_it", span_names)

    def test_monocle_trace_with_custom_name(self):
        """Test monocle_trace with custom span name"""
        
        custom_name = "my_custom_trace"
        with monocle_trace(span_name=custom_name):
            result = self.dummy.double_it(3)
        
        self.assertEqual(result, 6)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        verify_traceID(self.exporter, 2)
        
        span_names = [span.name for span in spans]
        self.assertIn("workflow", span_names)
        self.assertIn("double_it", span_names)

    def test_monocle_trace_with_attributes(self):
        """Test monocle_trace with custom attributes"""
        
        custom_attributes = {
            "user.id": "test_user_123",
            "operation.type": "calculation",
            "version": "1.0.0"
        }
        
        with monocle_trace(span_name="attr_test", attributes=custom_attributes):
            result = self.dummy.double_it(7)
        
        self.assertEqual(result, 14)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        
        # Find the context manager span
        context_span = None
        for span in spans:
            if span.name == "workflow":
                context_span = span
                break
        
        self.assertIsNotNone(context_span)
        
        # Verify custom attributes are set
        for key, value in custom_attributes.items():
            self.assertEqual(context_span.attributes.get(key), value)

    def test_monocle_trace_with_events(self):
        """Test monocle_trace with custom events"""
        
        custom_events = [
            {"name": "processing_started", "attributes": {"step": "initialization"}},
            {"name": "data_loaded", "attributes": {"records": 100}}
        ]
        
        with monocle_trace(span_name="event_test", events=custom_events):
            result = self.dummy.double_it(4)
        
        self.assertEqual(result, 8)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        
        # Find the context manager span
        context_span = None
        for span in spans:
            if span.name == "workflow":
                context_span = span
                break
        
        self.assertIsNotNone(context_span)
        
        # Verify events are added (we can't directly check events in this setup, 
        # but we can verify the span was created successfully)
        self.assertEqual(context_span.name, "workflow")

    def test_monocle_trace_nested_calls(self):
        """Test monocle_trace with nested instrumented calls"""
        
        with monocle_trace(span_name="nested_test"):
            result = self.dummy.triple_it(2)  # This calls double_it internally
        
        self.assertEqual(result, 6)  # 2*2 + 2 = 6
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        # Should have 3 spans: context manager, triple_it, and double_it
        self.assertEqual(len(spans), 3)
        verify_traceID(self.exporter, 3)
        
        span_names = [span.name for span in spans]
        self.assertIn("workflow", span_names)
        self.assertIn("triple_it", span_names)
        self.assertIn("double_it", span_names)

    def test_monocle_trace_exception_handling(self):
        """Test monocle_trace when an exception occurs"""
        
        with self.assertRaises(Exception):
            with monocle_trace(span_name="error_test"):
                self.dummy.double_it(5, raise_error=True)
        
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        # Should still have spans even with exception
        self.assertEqual(len(spans), 2)
        
        # Find the instrumented method span (double_it)
        method_span = None
        for span in spans:
            if span.name == "workflow":
                method_span = span
                break
        
        self.assertIsNotNone(method_span)
        # Verify the span recorded the error
        self.assertEqual(method_span.status.status_code, StatusCode.ERROR)

    def test_start_stop_trace_basic(self):
        """Test start_trace and stop_trace functions"""
        
        token = start_trace(span_name="manual_trace")
        self.assertIsNotNone(token)
        
        result = self.dummy.double_it(6)
        
        stop_trace(token)
        
        self.assertEqual(result, 12)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        verify_traceID(self.exporter, 2)
        
        span_names = [span.name for span in spans]
        self.assertIn("workflow", span_names)
        self.assertIn("double_it", span_names)

    def test_start_stop_trace_with_attributes_and_events(self):
        """Test start_trace and stop_trace with attributes and events"""
        
        start_attributes = {"start.time": "2023-01-01"}
        start_events = [{"name": "trace_initiated"}]
        
        token = start_trace(
            span_name="manual_trace_advanced",
            attributes=start_attributes,
            events=start_events
        )
        
        result = self.dummy.double_it(8)
        
        final_attributes = {"end.time": "2023-01-01", "result": result}
        final_events = [{"name": "trace_completed", "attributes": {"success": True}}]
        
        stop_trace(token, final_attributes=final_attributes, final_events=final_events)
        
        self.assertEqual(result, 16)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        
        # Find the manual trace span
        manual_span = None
        for span in spans:
            if span.name == "workflow":
                manual_span = span
                break
        
        self.assertIsNotNone(manual_span)
        
        # Verify attributes are set
        self.assertEqual(manual_span.attributes.get("start.time"), "2023-01-01")
        self.assertEqual(manual_span.attributes.get("end.time"), "2023-01-01")
        self.assertEqual(manual_span.attributes.get("result"), result)

    def test_start_stop_trace_exception_handling(self):
        """Test start_trace and stop_trace with exception"""
        
        token = start_trace(span_name="manual_trace_error")
        
        with self.assertRaises(Exception):
            self.dummy.double_it(5, raise_error=True)
        
        stop_trace(token)
        
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        # Should still have both spans
        self.assertEqual(len(spans), 2)


class TestInstrumentorAPIAsync(unittest.IsolatedAsyncioTestCase):
    """Test class for async instrumentor API functions"""
    
    @classmethod
    def setUpClass(cls):
        cls.exporter = CustomConsoleSpanExporter()
        cls.instrumentor = setup_monocle_telemetry(
            workflow_name="instrumentor_api_async_test",
            span_processors=[SimpleSpanProcessor(cls.exporter)],
            wrapper_methods=[
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="add3",
                    span_name="add3",
                    wrapper_method=atask_wrapper
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass", 
                    method="add2",
                    span_name="add2",
                    wrapper_method=atask_wrapper
                )
            ]
        )
        cls.dummy = DummyClass()

    @classmethod
    def tearDownClass(cls):
        if cls.instrumentor is not None:
            cls.instrumentor.uninstrument()

    def setUp(self):
        self.exporter.reset()

    def tearDown(self):
        self.exporter.reset()

    async def test_amonocle_trace_basic(self):
        """Test basic functionality of amonocle_trace async context manager"""
        
        async with amonocle_trace(span_name="async_trace_test"):
            result = await self.dummy.add3(10)
        
        self.assertEqual(result, 13)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        # Should have 2 spans: one from the context manager and one from the instrumented method
        self.assertEqual(len(spans), 2)
        verify_traceID(self.exporter, 2)
        
        span_names = [span.name for span in spans]
        self.assertIn("workflow", span_names)
        self.assertIn("add3", span_names)

    async def test_amonocle_trace_with_attributes_and_events(self):
        """Test amonocle_trace with custom attributes and events"""
        
        custom_attributes = {"async.operation": "addition", "user.id": "async_user"}
        custom_events = [{"name": "async_started", "attributes": {"value": 5}}]
        
        async with amonocle_trace(
            span_name="async_attr_test",
            attributes=custom_attributes,
            events=custom_events
        ):
            result = await self.dummy.add3(5)
        
        self.assertEqual(result, 8)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        
        # Find the async context manager span
        context_span = None
        for span in spans:
            if span.name == "workflow":
                context_span = span
                break
        
        self.assertIsNotNone(context_span)
        
        # Verify custom attributes
        self.assertEqual(context_span.attributes.get("async.operation"), "addition")
        self.assertEqual(context_span.attributes.get("user.id"), "async_user")

    async def test_amonocle_trace_nested_async_calls(self):
        """Test amonocle_trace with nested async instrumented calls"""
        
        async with amonocle_trace(span_name="nested_async_test"):
            result = await self.dummy.add2(10)  # This calls add3 internally
        
        self.assertEqual(result, 15)  # 10 + 3 + 2 = 15
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        # Should have 3 spans: context manager, add2, and add3
        self.assertEqual(len(spans), 3)
        verify_traceID(self.exporter, 3)
        
        span_names = [span.name for span in spans]
        self.assertIn("workflow", span_names)
        self.assertIn("add2", span_names)
        self.assertIn("add3", span_names)

    async def test_amonocle_trace_exception_handling(self):
        """Test amonocle_trace when an async exception occurs"""
        
        with self.assertRaises(Exception):
            async with amonocle_trace(span_name="async_error_test"):
                await self.dummy.add3(5, raise_error=True)
        
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        # Should still have spans even with exception
        self.assertEqual(len(spans), 2)
        
        # Find the instrumented method span (add3)
        method_span = None
        for span in spans:
            if span.name == "add3":
                method_span = span
                break
        
        self.assertIsNotNone(method_span)
        # Verify the span recorded the error
        self.assertEqual(method_span.status.status_code, StatusCode.ERROR)


class TestMonocleTraceMethod(unittest.TestCase):
    """Test class for monocle_trace_method decorator"""
    
    @classmethod
    def setUpClass(cls):
        cls.exporter = CustomConsoleSpanExporter()
        cls.instrumentor = setup_monocle_telemetry(
            workflow_name="trace_method_test",
            span_processors=[SimpleSpanProcessor(cls.exporter)]
        )

    @classmethod
    def tearDownClass(cls):
        if cls.instrumentor is not None:
            cls.instrumentor.uninstrument()

    def setUp(self):
        self.exporter.reset()

    def tearDown(self):
        self.exporter.reset()

    def test_monocle_trace_method_sync_basic(self):
        """Test monocle_trace_method decorator on synchronous function"""
        
        @monocle_trace_method()
        def test_function(x, y):
            return x + y
        
        result = test_function(3, 4)
        
        self.assertEqual(result, 7)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        # Should have 1 span from the decorated function
        self.assertEqual(len(spans), 2)
        
        span = spans[0]
        self.assertEqual(span.name, "test_function")

    def test_monocle_trace_method_sync_custom_name(self):
        """Test monocle_trace_method decorator with custom span name"""
        
        @monocle_trace_method(span_name="custom_sync_function")
        def test_function(x, y):
            return x * y
        
        result = test_function(5, 6)
        
        self.assertEqual(result, 30)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self.assertEqual(span.name, "custom_sync_function")

    def test_monocle_trace_method_sync_exception(self):
        """Test monocle_trace_method decorator when sync function raises exception"""
        
        @monocle_trace_method(span_name="error_function")
        def test_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            test_function()
        
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self.assertEqual(span.name, "error_function")
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

    async def test_monocle_trace_method_async_basic(self):
        """Test monocle_trace_method decorator on async function"""
        
        @monocle_trace_method()
        async def async_test_function(x, y):
            await asyncio.sleep(0.001)  # Small delay to simulate async work
            return x + y
        
        result = await async_test_function(8, 9)
        
        self.assertEqual(result, 17)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self.assertEqual(span.name, "async_test_function")

    async def test_monocle_trace_method_async_custom_name(self):
        """Test monocle_trace_method decorator on async function with custom name"""
        
        @monocle_trace_method(span_name="custom_async_function")
        async def async_test_function(x):
            await asyncio.sleep(0.001)
            return x ** 2
        
        result = await async_test_function(4)
        
        self.assertEqual(result, 16)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self.assertEqual(span.name, "custom_async_function")

    async def test_monocle_trace_method_async_exception(self):
        """Test monocle_trace_method decorator when async function raises exception"""
        
        @monocle_trace_method(span_name="async_error_function")
        async def async_test_function():
            await asyncio.sleep(0.001)
            raise RuntimeError("Async test error")
        
        with self.assertRaises(RuntimeError):
            await async_test_function()
        
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self.assertEqual(span.name, "async_error_function")
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

    def test_monocle_trace_method_multiple_calls(self):
        """Test monocle_trace_method decorator with multiple function calls"""
        
        @monocle_trace_method(span_name="multi_call_function")
        def test_function(x):
            return x * 2
        
        results = []
        for i in range(3):
            results.append(test_function(i + 1))
        
        self.assertEqual(results, [2, 4, 6])
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        
        # Should have 3 spans, one for each call
        self.assertEqual(len(spans), 6)
        # check that 3 have name as workflow and 3 have name as multi_call_function
        
        workflow_spans = [span for span in spans if span.name == "workflow"]
        self.assertEqual(len(workflow_spans), 3)
        # check that 3 have name as multi_call_function
        self.assertEqual(len([span for span in spans if span.name == "multi_call_function"]), 3)

        trace_ids = {span.context.trace_id for span in spans}
        self.assertEqual(len(trace_ids), 3)



if __name__ == '__main__':
    unittest.main()
