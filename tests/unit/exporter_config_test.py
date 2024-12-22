import logging
import os
import unittest

from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter

logger = logging.getLogger(__name__)

class TestHandler(unittest.TestCase):
    def test_default_exporter(self):
        default_exporter = get_monocle_exporter()
        assert default_exporter.__class__.__name__ == "FileSpanExporter"

    def test_fallback_exporter(self):
        """ No Okahu API key, it should fall back to console exporter"""
        os.environ["MONOCLE_EXPORTER"] = "okahu"
        default_exporter = get_monocle_exporter()
        assert default_exporter.__class__.__name__ == "ConsoleSpanExporter"

    def test_set_exporter(self):
        os.environ["MONOCLE_EXPORTER"] = "okahu"
        os.environ["OKAHU_API_KEY"] = "foo"
        default_exporter = get_monocle_exporter()
        assert default_exporter.__class__.__name__ == "OkahuSpanExporter"

if __name__ == "__main__":
    handler = TestHandler()
    handler.test_default_exporter()
    handler.test_fallback_exporter()
    handler.test_set_exporter()