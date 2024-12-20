import os
import unittest
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.common.constants import service_type_map, service_name_map
from dummy_class import DummyClass
from test_exporter import TestExporter

class TestHandler(unittest.TestCase):
    test_span_exporter = None

    def setUp(self, methodName='runTest'):
        app_name = "test"
        self.test_span_exporter = TestExporter()

        setup_monocle_telemetry(
            workflow_name=app_name,
            span_processors=[SimpleSpanProcessor(self.test_span_exporter)],
            wrapper_methods=[
                WrapperMethod(
                    package="dummy_class",
                    object_name="DummyClass",
                    method="dummy_chat",
                    span_name="langchain.workflow",
                    output_processor="output_processor",
                    wrapper_method=task_wrapper
                )
            ]
        )

    def test_codespaces(self):
        dummy_class_1 = DummyClass()

        for type_env, type_name in service_type_map.items():
            os.environ[type_env] = "true"

            entity_name_env = service_name_map.get(type_name)
            if entity_name_env is None:
                entity_name = "generic"
            else:
                entity_name = "test123"
                os.environ[entity_name_env] = entity_name

            self.test_span_exporter.set_trace_check({
                "entity.2.name": entity_name,
                "entity.2.type": "app_hosting." + type_name
            })

            dummy_class_1.dummy_chat("what is coffee?")

            del os.environ[type_env]
            if entity_name_env is not None and entity_name_env in os.environ:
                del os.environ[entity_name_env]

if __name__ == '__main__':
    unittest.main()
