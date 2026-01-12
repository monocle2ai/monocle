import logging
import os
import unittest
import warnings

from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter

logger = logging.getLogger(__name__)

class TestHandler(unittest.TestCase):
    def test_default_exporter(self):
        os.environ.clear()
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
        os.environ.clear()

    def test_set_exporter(self):
        os.environ["MONOCLE_EXPORTER"] = "okahu"
        os.environ["OKAHU_API_KEY"] = "foo"
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "OkahuSpanExporter"
        os.environ.clear()

    def test_memory_exporter(self):
        os.environ['MONOCLE_EXPORTER'] = "memory"
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "InMemorySpanExporter"
        os.environ.clear()

    def test_console_exporter(self):
        os.environ['MONOCLE_EXPORTER'] = "console"
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "ConsoleSpanExporter"
        os.environ.clear()

    def test_multi_exporter(self):
        os.environ['MONOCLE_EXPORTER'] = "file,memory,console"
        exporters = get_monocle_exporter()
        expected_exporters = ["FileSpanExporter", "InMemorySpanExporter", "ConsoleSpanExporter"]
        exporter_class_names = [exporter.__class__.__name__ for exporter in exporters]
        assert exporter_class_names == expected_exporters, f"Expected {expected_exporters}, but got {exporter_class_names}"
        os.environ.clear()

    def test_otlp_exporter(self):
        os.environ['MONOCLE_EXPORTER'] = "otlp"
        os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = "http://localhost:4318"
        default_exporter = get_monocle_exporter()
        assert default_exporter[0].__class__.__name__ == "OTLPSpanExporter"
        os.environ.clear()


if __name__ == "__main__":
    handler = TestHandler()
    handler.test_default_exporter()
    handler.test_fallback_exporter()
    handler.test_set_exporter()
    handler.test_memory_exporter()
    handler.test_console_exporter()
    handler.test_otlp_exporter()