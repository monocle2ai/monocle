import logging
import unittest
import asyncio
from common.dummy_class import DummyClass

from common.helpers import OurLLM
from common.http_span_exporter import HttpSpanExporter
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, monocle_trace_scope, start_scope, stop_scope
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from opentelemetry.trace.status import StatusCode

logger = logging.getLogger(__name__)
exporter = CustomConsoleSpanExporter()
setup_monocle_telemetry(
            workflow_name="async_test",
            span_processors=[
                    SimpleSpanProcessor(exporter)
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

class TestHandler(unittest.IsolatedAsyncioTestCase):
    dummy = DummyClass()
    
    def setUp(self):
        exporter.reset()

    def tearDown(self):
        exporter.reset()
        return super().tearDown()
    
    # verify nested async calls have same traceID
    async def test_nested_async_traceID(self):
        res = await self.dummy.add1(10)
        assert res == 16
        exporter.force_flush()
        spans = exporter.captured_spans
        assert len(spans) == 4
        traceID = None
        for span in spans:
            if traceID == None:
                traceID = span.context.trace_id
            else:
                assert traceID == span.context.trace_id

    # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_API(self):
        SCOPE_NAME="test_scope1"
        SCOPE_VALUE="test1"
        token = start_scope(SCOPE_NAME, SCOPE_VALUE)
        res = await self.dummy.add1(10)
        stop_scope(token)
        assert res == 16
        exporter.force_flush()
        spans = exporter.captured_spans
        assert len(spans) == 4
        for span in spans:
            assert span.attributes.get("scope."+SCOPE_NAME) == SCOPE_VALUE

    # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_wrapper(self):
        SCOPE_NAME="test_scope1"
        SCOPE_VALUE="test1"
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            res = await self.dummy.add1(10)
        assert res == 16
        exporter.force_flush()
        spans = exporter.captured_spans
        assert len(spans) == 4
        for span in spans:
            assert span.attributes.get("scope."+SCOPE_NAME) == SCOPE_VALUE

    # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_wrapper_errors(self):
        SCOPE_NAME="test_scope1"
        SCOPE_VALUE="test1"
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            try:
                res = await self.dummy.add1(10,raise_error=True)
                assert False
            except Exception as e:
                print(f"Got exception {e}")
        exporter.force_flush()
        spans = exporter.captured_spans
        assert len(spans) == 4
        traceID = None
        for span in spans:
            assert span.attributes.get("scope."+SCOPE_NAME) == SCOPE_VALUE
            assert span.status.status_code == StatusCode.ERROR
            if traceID == None:
                traceID = span.context.trace_id
            else:
                assert traceID == span.context.trace_id

if __name__ == '__main__':
    unittest.main()