import os
import logging
import unittest
from common.dummy_class import DummyClass
from common.utils import verify_scope, verify_traceID, SCOPE_NAME, SCOPE_VALUE, MULTIPLE_SCOPES, verify_multiple_scopes, get_scope_values_from_args
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, monocle_trace_scope, start_scope, stop_scope, monocle_trace_scope_method
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, scopes_wrapper
from opentelemetry.trace.status import StatusCode

logger = logging.getLogger(__name__)
exporter = CustomConsoleSpanExporter()
class TestHandler(unittest.IsolatedAsyncioTestCase):
    dummy = DummyClass()
    
    def setUp(self):
        exporter.reset()

    def tearDown(self):
        exporter.reset()
        return super().tearDown()
    
    @classmethod
    def setUpClass(cls):
        exporter.reset()
        setup_monocle_telemetry(
            workflow_name="async_test",
            span_processors=[
                    SimpleSpanProcessor(exporter)
                ],
            wrapper_methods=[
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="double_it",
                    span_name="double_it",
                    wrapper_method= task_wrapper
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="triple_it",
                    span_name="triple_it",
                    wrapper_method= task_wrapper
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="scope_values_test_method",
                    span_name="scope_values_test_method",
                    scope_values=lambda args, kwargs: MULTIPLE_SCOPES,
                    wrapper_method=scopes_wrapper
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="scope_values_dynamic_test_method",
                    span_name="scope_values_dynamic_test_method",
                    scope_values=get_scope_values_from_args,
                    wrapper_method=scopes_wrapper
                )
        ])
        return super().setUpClass()

    # verify nested instrumented calls have same traceID
    def test_nested_traceID(self):
        res = self.dummy.triple_it(10)
        assert res == 30
        verify_traceID(exporter, excepted_span_count=3)

    # verify nested async calls have same scope set using scope API
    def test_nested_scopes_with_API(self):
        token = start_scope(SCOPE_NAME, SCOPE_VALUE)
        res = self.dummy.triple_it(10)
        stop_scope(token)
        assert res == 30
        verify_scope(exporter, excepted_span_count=3)

    # verify nested async calls have same scope set using scope API
    def test_nested_scopes_with_wrapper(self):
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            res = self.dummy.triple_it(10)
            assert res == 30
        verify_scope(exporter, excepted_span_count=3)

    # verify nested async calls have same scope set using scope API
    def test_nested_scopes_with_wrapper_errors(self):
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            try:
                res = self.dummy.triple_it(10, raise_error=True)
                assert False
            except Exception as e:
                print(f"Got exception {e}")
        exporter.force_flush()
        spans = exporter.captured_spans
        assert len(spans) == 3
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
    def test_nested_scopes_with_decorator(self):
        res = self.dummy.scope_decorator_test_method()
        assert res == 30
        verify_scope(exporter, excepted_span_count=3)

    # verify nested async calls have same scope set using scope decorator
    async def test_nested_scopes_with_config(self):
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            res = self.dummy.triple_it(10)
            assert res == 30

        verify_scope(exporter, excepted_span_count=3)
        
    # verify nested calls have same scopes set using scope_values
    def test_nested_scopes_with_scope_values(self):
        res = self.dummy.scope_values_test_method()
        assert res == 30
        verify_multiple_scopes(exporter, MULTIPLE_SCOPES, excepted_span_count=3)
        
    # verify nested calls with dynamic scope values extracted from args
    def test_nested_scopes_with_dynamic_scope_values(self):
        user_id = "user123"
        session_id = "session456"
        
        expected_scopes = {
            "user_id": user_id,
            "session_id": session_id,
        }
        
        res = self.dummy.scope_values_dynamic_test_method(user_id, session_id)
        assert res == 30
        verify_multiple_scopes(exporter, expected_scopes, excepted_span_count=3)

if __name__ == '__main__':
    unittest.main()
