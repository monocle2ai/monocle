import logging
import os
import unittest

from base_unit import MonocleTestBase
from common.dummy_class import DummyClass
from common.mock_span_exporter import MockSpanExporter
from monocle_apptrace.instrumentation.common.constants import (
    service_name_map,
    service_type_map,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = logging.getLogger(__name__)

class TestHandler(MonocleTestBase):

    def test_codespaces(self):
        for type_env, type_name in service_type_map.items():
            with self.subTest(service_type=type_name):
                # Clean environment first
                for key in list(os.environ.keys()):
                    if key in service_type_map or key in service_name_map.values():
                        del os.environ[key]
                
                # Set up environment variables for this test case
                os.environ[type_env] = "true"

                entity_name_env = service_name_map.get(type_name)
                if entity_name_env is None:
                    entity_name = "generic"
                else:
                    entity_name = "test123"
                    os.environ[entity_name_env] = entity_name

                # Create fresh instrumentor for each test case with proper teardown
                app_name = "test"
                with MockSpanExporter() as test_span_exporter:
                    instrumentor = setup_monocle_telemetry(
                        workflow_name=app_name,
                        span_processors=[SimpleSpanProcessor(test_span_exporter)],
                        wrapper_methods=[
                            WrapperMethod(
                                package="common.dummy_class",
                                object_name="DummyClass",
                                method="dummy_chat",
                                span_name="langchain.workflow",
                                output_processor="output_processor",
                                wrapper_method=task_wrapper
                            )
                        ]
                    )

                    try:
                        test_span_exporter.set_trace_check({
                            "entity.2.name": entity_name,
                            "entity.2.type": "app_hosting." + type_name
                        })

                        dummy_class_1 = DummyClass()
                        dummy_class_1.dummy_chat("what is coffee?")

                    finally:
                        # Clean up instrumentor
                        try:
                            instrumentor.uninstrument()
                        except Exception as e:
                            logger.info("Uninstrument failed:", e)
                    
                    # Clean up environment variables
                    if type_env in os.environ:
                        del os.environ[type_env]
                    if entity_name_env is not None and entity_name_env in os.environ:
                        del os.environ[entity_name_env]

if __name__ == '__main__':
    unittest.main()
