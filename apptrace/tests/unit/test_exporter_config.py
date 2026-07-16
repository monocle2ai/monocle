import logging
import os
import unittest
import warnings
from unittest.mock import patch

from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter

logger = logging.getLogger(__name__)


class TestHandler(unittest.TestCase):
    def setUp(self):
        # Isolate each test from ambient env vars and .env file lookups to prevent config leakage.
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()
        self._env_value_patcher = patch(
            "monocle_apptrace.exporters.okahu.okahu_exporter.get_monocle_env_value",
            side_effect=lambda key: os.environ.get(key),
        )
        self._env_value_patcher.start()

    def tearDown(self):
        self._env_value_patcher.stop()
        self._env_patcher.stop()

    def test_default_exporter(self):
        """No configuration -> defaults to FileSpanExporter."""
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "FileSpanExporter"

    def test_fallback_exporter(self):
        """ No Okahu API key, it should fall back to console exporter"""
        os.environ["MONOCLE_EXPORTER"] = "okahu"
        # Suppress expected warning about missing OKAHU_API_KEY during fallback test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "ConsoleSpanExporter"

    def test_set_exporter(self):
        os.environ["MONOCLE_EXPORTER"] = "okahu"
        os.environ["OKAHU_API_KEY"] = "foo"
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "OkahuSpanExporter"

    def test_memory_exporter(self):
        os.environ['MONOCLE_EXPORTER'] = "memory"
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "MonocleInMemorySpanExporter"

    def test_console_exporter(self):
        os.environ['MONOCLE_EXPORTER'] = "console"
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "ConsoleSpanExporter"

    def test_multi_exporter(self):
        os.environ['MONOCLE_EXPORTER'] = "file,memory,console"
        exporters = get_monocle_exporter()
        expected_exporters = ["FileSpanExporter", "MonocleInMemorySpanExporter", "ConsoleSpanExporter"]
        exporter_class_names = [exporter.__class__.__name__ for exporter in exporters]
        assert exporter_class_names == expected_exporters, f"Expected {expected_exporters}, but got {exporter_class_names}"

    def test_otlp_exporter(self):
        os.environ['MONOCLE_EXPORTER'] = "otlp"
        os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = "http://localhost:4318"
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "OTLPSpanExporter"

    def test_otlp_genai_semconv_exporter(self):
        os.environ['MONOCLE_EXPORTER'] = "otlp-genai-semconv"
        os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = "http://localhost:4318"
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "OTLPSpanExporter"

    def test_otlp_exporter_supports_authenticated_headers(self):
        with patch.dict(
            os.environ,
            {
                "MONOCLE_EXPORTER": "otlp",
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://example.test/v1/traces",
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS": (
                    "Authorization=Bearer%20test-token,X-Tenant-ID=test-tenant"
                ),
            },
            clear=True,
        ):
            exporter = get_monocle_exporter()[0]

            assert exporter._endpoint == "https://example.test/v1/traces"
            assert exporter._session.headers["authorization"] == "Bearer test-token"
            assert exporter._session.headers["x-tenant-id"] == "test-tenant"


    def test_monocle_console_adds_console_exporter(self):
        """MONOCLE_CONSOLE=true appends ConsoleSpanExporter alongside the primary exporter."""
        os.environ["MONOCLE_EXPORTER"] = "file"
        os.environ["MONOCLE_CONSOLE"] = "true"
        exporters = get_monocle_exporter()
        class_names = [e.__class__.__name__ for e in exporters]
        assert "FileSpanExporter" in class_names
        assert "ConsoleSpanExporter" in class_names

    def test_monocle_console_no_duplicate(self):
        """MONOCLE_CONSOLE=true does not add a second ConsoleSpanExporter when console is already configured."""
        os.environ["MONOCLE_EXPORTER"] = "console"
        os.environ["MONOCLE_CONSOLE"] = "true"
        exporters = get_monocle_exporter()
        console_count = sum(1 for e in exporters if e.__class__.__name__ == "ConsoleSpanExporter")
        assert console_count == 1

    def test_monocle_console_unset_no_effect(self):
        """Without MONOCLE_CONSOLE, no extra ConsoleSpanExporter is added."""
        os.environ["MONOCLE_EXPORTER"] = "file"
        exporters = get_monocle_exporter()
        class_names = [e.__class__.__name__ for e in exporters]
        assert "ConsoleSpanExporter" not in class_names


if __name__ == "__main__":
    unittest.main()
