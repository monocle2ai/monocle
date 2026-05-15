import os
import logging
import unittest
from common.dummy_class import DummyClass
from common.utils import verify_scope, verify_traceID, SCOPE_NAME, SCOPE_VALUE
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace
from monocle_apptrace import setup_monocle_telemetry, monocle_trace_scope, start_scope, stop_scope, monocle_trace_scope_method
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from opentelemetry.trace.status import StatusCode

logger = logging.getLogger(__name__)

class TestHandler(unittest.IsolatedAsyncioTestCase):
    dummy = DummyClass()
    instrumentor = None
    exporter = None
    
    def setUp(self):
        #  Clear any existing instrumentation state
        from monocle_apptrace.instrumentation.common.instrumentor import get_monocle_instrumentor, set_monocle_instrumentor, set_monocle_setup_signature
        existing = get_monocle_instrumentor()
        if existing is not None:
            try:
                existing.uninstrument()
            except:
                pass
        set_monocle_instrumentor(None)
        set_monocle_setup_signature(None)
        
        # Create new exporter and instrumentor for each test
        self.exporter = CustomConsoleSpanExporter()
        self.exporter.reset()
        
        self.instrumentor = setup_monocle_telemetry(
            workflow_name="async_test",
            span_processors=[
                    SimpleSpanProcessor(self.exporter)
                ],
            wrapper_methods=[
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="add1",
                    span_name="add1",
                    wrapper_method= atask_wrapper
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="add2",
                    span_name="add2",
                    wrapper_method= atask_wrapper
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="add3", 
                    span_name="add3",
                    wrapper_method= atask_wrapper
                )                        
        ])

    def tearDown(self):
        try:
            # Shutdown tracer provider to flush and close all span processors
            trace.get_tracer_provider().shutdown()
            
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            logger.warning(f"TearDown failed: {e}")
        
        self.exporter = None
        self.instrumentor = None
        return super().tearDown()

    # verify nested async calls have same traceID
    async def test_nested_async_traceID(self):
        res = await self.dummy.add1(10)
        assert res == 16
        # Filter out Haystack spans
        self.exporter.force_flush()
        app_spans = [s for s in self.exporter.captured_spans if "haystack" not in s.name.lower()]
        self.exporter.captured_spans = app_spans
        verify_traceID(self.exporter, excepted_span_count=4)

    # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_API(self):
        token = start_scope(SCOPE_NAME, SCOPE_VALUE)
        res = await self.dummy.add1(10)
        stop_scope(token)
        assert res == 16
        # Filter out Haystack autoenable spans that may occur during setup
        self.exporter.force_flush()
        app_spans = [s for s in self.exporter.captured_spans if "haystack" not in s.name.lower()]
        self.exporter.captured_spans = app_spans
        verify_scope(self.exporter, excepted_span_count=4)

    # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_wrapper(self):
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            res = await self.dummy.add1(10)
        assert res == 16
        # Filter out Haystack spans
        self.exporter.force_flush()
        app_spans = [s for s in self.exporter.captured_spans if "haystack" not in s.name.lower()]
        self.exporter.captured_spans = app_spans
        verify_scope(self.exporter, excepted_span_count=4)

    # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_wrapper_errors(self):
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            try:
                res = await self.dummy.add1(10,raise_error=True)
                assert False
            except Exception as e:
                logger.info(f"Got exception {e}")
        self.exporter.force_flush()
        app_spans = [s for s in self.exporter.captured_spans if "haystack" not in s.name.lower()]
        spans = app_spans
        assert len(spans) == 4
        traceID = None
        for span in spans:
            assert span.attributes.get("scope."+SCOPE_NAME) == SCOPE_VALUE
            if span.attributes.get("span.type") != "workflow":
                assert span.status.status_code == StatusCode.ERROR
            if traceID == None:
                traceID = span.context.trace_id
            else:
                assert traceID == span.context.trace_id

    # verify nested async calls have same scope set using scope decorator
    async def test_nested_async_scopes_with_decorator(self):
        res = await self.dummy.scope_async_decorator_test_method()
        assert res == 16
        # Filter out Haystack spans
        self.exporter.force_flush()
        app_spans = [s for s in self.exporter.captured_spans if "haystack" not in s.name.lower()]
        self.exporter.captured_spans = app_spans
        verify_scope(self.exporter, excepted_span_count=4)

    # verify nested async calls have same scope set using scope decorator
    async def test_nested_async_scopes_with_config(self):
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            res = await self.dummy.scope_async_config_test_method()
        assert res == 16
        # Filter out Haystack spans
        self.exporter.force_flush()
        app_spans = [s for s in self.exporter.captured_spans if "haystack" not in s.name.lower()]
        self.exporter.captured_spans = app_spans
        verify_scope(self.exporter, excepted_span_count=4)

if __name__ == '__main__':
    unittest.main()