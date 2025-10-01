import json
import logging
import os
import time
import unittest

from common.dummy_class import DummyClass, dummy_wrapper
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod

logger = logging.getLogger(__name__)

class TestHandler(unittest.TestCase):
    span_processor = None
    file_exporter = None

    def tearDown(self) -> None:
        try:
            self.span_processor.shutdown()
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            print("Uninstrument failed:", e)
        return super().tearDown()

    def setUp(self):
        app_name = "file_test"
        self.file_exporter = FileSpanExporter(time_format="%Y-%m-%d")
        self.span_processor = SimpleSpanProcessor(self.file_exporter)
        self.instrumentor = setup_monocle_telemetry(
            workflow_name=app_name,
            span_processors=[
                    self.span_processor
            ],
            union_with_default_methods=False,
            wrapper_methods=[
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="dummy_method",
                    span_name="dummy.span",
                    wrapper_method=dummy_wrapper)
            ])

    def test_file_exporter(self):
        
        dummy_class_1 = DummyClass()
        dummy_class_1.dummy_method()

        self.span_processor.force_flush()
        
        trace_file_name = self.file_exporter.last_file_processed
        if trace_file_name is None:
            print("Inside no file")
            time.sleep(10)
        else:
            print("file name is : ",trace_file_name)

        try:
            with open(trace_file_name) as f:
                print("Inside file")
                trace_data = json.load(f)
                trace_id_from_file = trace_data[0]["context"]["trace_id"]
                trace_id_from_exporter = hex(self.file_exporter.last_trace_id)
                assert trace_id_from_file == trace_id_from_exporter

            os.remove(trace_file_name)
        except Exception as ex:
            print("Got error " + str(ex))
            assert False
       

if __name__ == '__main__':
    unittest.main()