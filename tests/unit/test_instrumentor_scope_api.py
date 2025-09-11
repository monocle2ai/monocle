import unittest
import asyncio
from common.dummy_class import DummyClass
from common.utils import verify_scope, SCOPE_NAME, SCOPE_VALUE, MULTIPLE_SCOPES
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace import (
    setup_monocle_telemetry,
    start_scope,
    stop_scope,
    monocle_trace_scope,
    amonocle_trace_scope,
    monocle_trace_scope_method
)
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper

class TestInstrumentorScopeAPI(unittest.TestCase):
    """Test class for scope API functions like start_scope, stop_scope, monocle_trace_scope, monocle_trace_scope_method"""

    @classmethod
    def setUpClass(cls):
        cls.exporter = CustomConsoleSpanExporter()
        cls.instrumentor = setup_monocle_telemetry(
            workflow_name="instrumentor_scope_api_test",
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

    def test_start_stop_scope_basic(self):
        """Test start_scope and stop_scope functions"""
        token = start_scope(SCOPE_NAME, SCOPE_VALUE)
        result = self.dummy.double_it(5)
        stop_scope(token)
        self.assertEqual(result, 10)
        self.exporter.force_flush()
        verify_scope(self.exporter, excepted_span_count=2)

    def test_start_stop_scope_nested(self):
        """Test nested start_scope and stop_scope"""
        token1 = start_scope(SCOPE_NAME, SCOPE_VALUE)
        token2 = start_scope("nested_scope", "nested_value")
        result = self.dummy.double_it(7)
        stop_scope(token2)
        stop_scope(token1)
        self.assertEqual(result, 14)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        self.assertEqual(len(spans), 2)
        # Check both scopes are present
        span = spans[0]
        self.assertEqual(span.attributes.get(f"scope.{SCOPE_NAME}"), SCOPE_VALUE)
        self.assertEqual(span.attributes.get("scope.nested_scope"), "nested_value")

    def test_monocle_trace_scope_context_manager(self):
        """Test monocle_trace_scope context manager"""
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            result = self.dummy.double_it(3)
        self.assertEqual(result, 6)
        self.exporter.force_flush()
        verify_scope(self.exporter, excepted_span_count=2)

    def test_monocle_trace_scope_nested(self):
        """Test nested monocle_trace_scope context managers"""
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            with monocle_trace_scope("nested_scope", "nested_value"):
                result = self.dummy.double_it(4)
        self.assertEqual(result, 8)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self.assertEqual(span.attributes.get(f"scope.{SCOPE_NAME}"), SCOPE_VALUE)
        self.assertEqual(span.attributes.get("scope.nested_scope"), "nested_value")

    def test_monocle_trace_scope_method_decorator(self):
        """Test monocle_trace_scope_method decorator on sync function"""
        @monocle_trace_scope_method(SCOPE_NAME, SCOPE_VALUE)
        def test_func(x):
            return self.dummy.double_it(x)
        result = test_func(9)
        self.assertEqual(result, 18)
        self.exporter.force_flush()
        verify_scope(self.exporter, excepted_span_count=2)

    def test_monocle_trace_scope_method_multiple_calls(self):
        """Test monocle_trace_scope_method with multiple calls"""
        @monocle_trace_scope_method(SCOPE_NAME, SCOPE_VALUE)
        def test_func(x):
            return self.dummy.double_it(x)
        results = [test_func(i) for i in range(1, 4)]
        self.assertEqual(results, [2, 4, 6])
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        self.assertEqual(len(spans), 6)
        for span in spans:
            self.assertEqual(span.attributes.get(f"scope.{SCOPE_NAME}"), SCOPE_VALUE)

import pytest

@pytest.mark.asyncio
class TestInstrumentorScopeAPIAsync(unittest.IsolatedAsyncioTestCase):
    """Test class for async scope API functions"""
    @classmethod
    def setUpClass(cls):
        cls.exporter = CustomConsoleSpanExporter()
        cls.instrumentor = setup_monocle_telemetry(
            workflow_name="instrumentor_scope_api_async_test",
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

    async def test_amonocle_trace_scope_async_context_manager(self):
        async with amonocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            result = await self.dummy.add3(10)
        self.assertEqual(result, 13)
        self.exporter.force_flush()
        verify_scope(self.exporter, excepted_span_count=2)

    async def test_amonocle_trace_scope_nested(self):
        async with amonocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            async with amonocle_trace_scope("nested_scope", "nested_value"):
                result = await self.dummy.add3(5)
        self.assertEqual(result, 8)
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        self.assertEqual(len(spans), 2)
        span = spans[0]
        self.assertEqual(span.attributes.get(f"scope.{SCOPE_NAME}"), SCOPE_VALUE)
        self.assertEqual(span.attributes.get("scope.nested_scope"), "nested_value")

    async def test_monocle_trace_scope_method_async(self):
        @monocle_trace_scope_method(SCOPE_NAME, SCOPE_VALUE)
        async def test_func(x):
            return await self.dummy.add3(x)
        result = await test_func(7)
        self.assertEqual(result, 10)
        self.exporter.force_flush()
        verify_scope(self.exporter, excepted_span_count=2)

    async def test_monocle_trace_scope_method_async_multiple_calls(self):
        @monocle_trace_scope_method(SCOPE_NAME, SCOPE_VALUE)
        async def test_func(x):
            return await self.dummy.add3(x)
        results = [await test_func(i) for i in range(1, 4)]
        self.assertEqual(results, [4, 5, 6])
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        self.assertEqual(len(spans), 6)
        for span in spans:
            self.assertEqual(span.attributes.get(f"scope.{SCOPE_NAME}"), SCOPE_VALUE)

if __name__ == '__main__':
    unittest.main()
