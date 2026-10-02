import logging
import os
import unittest
from unittest.mock import patch

from base_unit import MonocleTestBase
from common.dummy_class import DummyClass
from common.mock_span_exporter import MockSpanExporter
from monocle_apptrace.instrumentation.common.constants import (
    GCP_AGENT_ENGINE_ENV_NAME,
    GCP_AGENT_ENGINE_LOCATION_ENV_NAME,
    GCP_PROJECT_ENV_NAME,
    service_name_map,
    service_type_map,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.instrumentor import get_monocle_instrumentor
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

    def _hosting_entity(self, env):
        """Run one traced call with only ``env`` as hosting variables and return the
        workflow span's (entity.2.type, entity.2.name)."""
        hosting_keys = set(service_type_map) | set(service_name_map.values()) | {
            GCP_PROJECT_ENV_NAME, GCP_AGENT_ENGINE_LOCATION_ENV_NAME, "GOOGLE_CLOUD_LOCATION", "K_SERVICE"}
        with patch.dict(os.environ):
            for key in hosting_keys:
                os.environ.pop(key, None)
            os.environ.update(env)

            # monocle_test_tools' plugin sets up telemetry after each subtest; clear it
            # so this setup is not skipped as a duplicate.
            existing = get_monocle_instrumentor()
            if existing is not None and existing.is_instrumented_by_opentelemetry:
                existing.uninstrument()

            with MockSpanExporter() as test_span_exporter:
                instrumentor = setup_monocle_telemetry(
                    workflow_name="test",
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
                    DummyClass().dummy_chat("what is coffee?")
                finally:
                    instrumentor.uninstrument()
                spans = [span for batch in test_span_exporter.get_exported_spans() for span in batch["batch"]]

        workflow_spans = [s for s in spans if s["attributes"].get("span.type") == "workflow"]
        self.assertEqual(len(workflow_spans), 1, f"expected one workflow span, got {len(workflow_spans)}")
        attributes = workflow_spans[0]["attributes"]
        return attributes.get("entity.2.type"), attributes.get("entity.2.name")

    def test_agent_engine_full_resource_name(self):
        self.assertEqual(
            self._hosting_entity({
                GCP_AGENT_ENGINE_ENV_NAME: "1234567890123456789",
                GCP_AGENT_ENGINE_LOCATION_ENV_NAME: "us-central1",
                GCP_PROJECT_ENV_NAME: "123456789012",
            }),
            (
                "app_hosting.gcp_agent_engine",
                "projects/123456789012/locations/us-central1/reasoningEngines/1234567890123456789",
            ),
        )

    def test_agent_engine_missing_project_or_location_uses_engine_id(self):
        cases = {
            "no project": {GCP_AGENT_ENGINE_LOCATION_ENV_NAME: "us-central1"},
            "no location": {GCP_PROJECT_ENV_NAME: "123456789012"},
            "neither": {},
            # The resource name uses the engine's own location, not the general one.
            "only generic location": {GCP_PROJECT_ENV_NAME: "123456789012", "GOOGLE_CLOUD_LOCATION": "us-central1"},
        }
        for case, env in cases.items():
            with self.subTest(case=case):
                self.assertEqual(
                    self._hosting_entity({GCP_AGENT_ENGINE_ENV_NAME: "1234567890123456789", **env}),
                    ("app_hosting.gcp_agent_engine", "1234567890123456789"),
                )

    def test_plain_gcp_environment_is_not_agent_engine(self):
        self.assertEqual(
            self._hosting_entity({
                GCP_PROJECT_ENV_NAME: "my-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
                GCP_AGENT_ENGINE_LOCATION_ENV_NAME: "us-central1",
                "K_SERVICE": "some-cloud-run-service",
            }),
            ("app_hosting.generic", "generic"),
        )

    def test_agent_engine_takes_precedence_over_cloud_run(self):
        self.assertEqual(
            self._hosting_entity({
                GCP_AGENT_ENGINE_ENV_NAME: "1234567890123456789",
                GCP_AGENT_ENGINE_LOCATION_ENV_NAME: "us-central1",
                GCP_PROJECT_ENV_NAME: "123456789012",
                "K_SERVICE": "reasoning-engine-1234567890123456789",
            }),
            (
                "app_hosting.gcp_agent_engine",
                "projects/123456789012/locations/us-central1/reasoningEngines/1234567890123456789",
            ),
        )

    def test_agent_engine_is_checked_first(self):
        """Detection is first match in map order; Agent Engine must lead it."""
        self.assertEqual(next(iter(service_type_map)), GCP_AGENT_ENGINE_ENV_NAME)

if __name__ == '__main__':
    unittest.main()
