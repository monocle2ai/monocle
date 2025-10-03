import unittest
import threading
import time
from flask import Flask
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from common.dummy_class import DummyClass, dummy_wrapper
from common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry import trace
import logging


logger = logging.getLogger(__name__)
class TestValidateResponseMultithreaded(unittest.TestCase):

    def setUp(self):
        self.app = Flask(__name__)
        FlaskInstrumentor().instrument_app(app=self.app, enable_commenter=True, commenter_options={})

        @self.app.route("/")
        def hello_world():
            app_name = "test"
            dummy_class_1 = DummyClass()
            dummy_class_1.dummy_method()
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("child_span") as child_span:
                return "<p>Hello, World!</p>"

        # Set up telemetry
        self.capturing_exporter = CustomConsoleSpanExporter()
        span_processor = SimpleSpanProcessor(self.capturing_exporter)
        self.instrumentor = setup_monocle_telemetry(
            workflow_name="test_1",
            span_processors=[span_processor],
            wrapper_methods=[
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="dummy_method",
                    span_name="langchain.workflow",
                    wrapper_method=dummy_wrapper
                )
            ],
        )

    def tearDown(self) -> None:
        try:
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            logger.info("Uninstrument failed:", e)
        return super().tearDown()

    @staticmethod
    def send_request(app, headers, captured_responses):
        client = app.test_client()
        response = client.get("/", headers=headers)
        captured_responses.append(response)
        time.sleep(3)

    def test_validate_response_multithreaded(self):
        trace_id = "0af7651916cd43dd8448eb211c80319c"
        parent_id = "b7ad6b7169203331"
        traceparent = f"00-{trace_id}-{parent_id}-01"
        headers = {"traceparent": traceparent}

        num_threads = 3
        threads = []
        captured_responses = []

        # Create threads to send requests
        for _ in range(num_threads):
            thread = threading.Thread(target=self.send_request, args=(self.app, headers, captured_responses))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Validate the responses
        for response in captured_responses:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data.decode("utf-8"), "<p>Hello, World!</p>")
            self.assertEqual(response.request.headers.get("traceparent"), traceparent)

        # Validate the spans
        spans = self.capturing_exporter.captured_spans
        self.assertGreater(len(spans), 0, "No spans were captured.")

        root_spans = []
        child_spans = []
        grand_child_spans = []

        for span in spans:
            if span.name == "GET /":
                root_spans.append(span)
            elif span.name == "langchain.workflow":
                child_spans.append(span)
            elif span.name == "child_span":
                grand_child_spans.append(span)

        for root_span in root_spans:
            root_trace_id = f"{root_span.context.trace_id:032x}"
            root_parent_id = f"{root_span.parent.span_id:016x}"
            self.assertEqual(root_trace_id, trace_id)
            self.assertEqual(root_parent_id, parent_id)

        for child_span in child_spans:
            child_trace_id = f"{child_span.context.trace_id:032x}"
            child_parent_id = f"{child_span.parent.span_id:016x}"
            matching_root_span = next(
                (root for root in root_spans if f"{root.context.span_id:016x}" == child_parent_id),
                None,
            )
            matching_root_span_id = f"{matching_root_span.context.span_id:016x}"
            self.assertEqual(child_parent_id, matching_root_span_id)
            self.assertEqual(child_trace_id, trace_id)

        for grandchild_span in grand_child_spans:
            grandchild_trace_id = f"{grandchild_span.context.trace_id:032x}"
            grandchild_parent_id = f"{grandchild_span.parent.span_id:016x}"

            matching_child_span = next(
                (child for child in child_spans if f"{child.parent.span_id:016x}" == grandchild_parent_id),
                None,
            )
            matching_child_span_id = f"{matching_child_span.parent.span_id:016x}"
            self.assertEqual(grandchild_parent_id, matching_child_span_id)
            self.assertEqual(grandchild_trace_id, trace_id, "Trace IDs do not match.")

        self.assertEqual(len(root_spans), num_threads)
        self.assertEqual(len(child_spans), num_threads)
        self.assertEqual(len(grand_child_spans), num_threads)

if __name__ == "__main__":
    unittest.main()
