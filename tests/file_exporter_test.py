import json
import logging
import os
import unittest
from dummy_class import DummyClass, dummy_wrapper

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler('traces.txt','w')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)


class TestHandler(unittest.TestCase):
    SPAN_NAME="dummy.span"
    def test_file_exporter(self):
        app_name = "file_test"
        file_exporter = FileSpanExporter(time_format="%Y-%m-%d")
        span_processor = BatchSpanProcessor(file_exporter)
        setup_monocle_telemetry(
            workflow_name=app_name,
            span_processors=[
                    span_processor
            ],
            wrapper_methods=[
                WrapperMethod(
                    package="dummy_class",
                    object_name="DummyClass",
                    method="dummy_method",
                    span_name=self.SPAN_NAME,
                    wrapper_method=dummy_wrapper)
            ])
        dummy_class_1 = DummyClass()

        dummy_class_1.dummy_method()

        span_processor.force_flush()
        span_processor.shutdown()
        trace_file_name = file_exporter.current_file_path

        try:
            with open(trace_file_name) as f:
                trace_data = json.load(f)
                trace_id_from_file = trace_data["context"]["trace_id"]
                trace_id_from_exporter = hex(file_exporter.current_trace_id)
                assert trace_id_from_file == trace_id_from_exporter

                span_name = trace_data["name"]
                assert self.SPAN_NAME == span_name

            os.remove(trace_file_name)
        except Exception as ex:
            print("Got error " + str(ex))
            assert False

