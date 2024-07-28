import json
import logging
import os
import unittest
from dummy_class import DummyClass, dummy_wrapper

from monocle_apptrace.instrumentor import setup_monocle_telemetry
from monocle_apptrace.wrapper import WrapperMethod
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
                    object="DummyClass",
                    method="dummy_method",
                    span_name=self.SPAN_NAME,
                    wrapper=dummy_wrapper)
            ])
        dummy_class_1 = DummyClass()

        dummy_class_1.dummy_method()

        span_processor.force_flush()
        span_processor.shutdown()
        trace_file_name = file_exporter.current_file_path
        with open(trace_file_name) as f:
            trace_data = json.load(f)
            trace_id_from_file = trace_data["context"]["trace_id"]
            trace_id_from_exporter = hex(file_exporter.current_trace_id)
            assert trace_id_from_file == trace_id_from_exporter

            span_name = trace_data["name"]
            assert self.SPAN_NAME == span_name

        os.remove(trace_file_name)
