"""Providers are found by name, and core exporter lookup goes through them."""
import os
import unittest
from unittest.mock import patch

from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter
from monocle_apptrace.providers import Provider, get_provider, provider_names, register_provider
from monocle_apptrace.providers import registry


class _MemoryProvider(Provider):
    name = "test-memory-provider"
    span_exporter = {"module": "monocle_apptrace.exporters.base_exporter",
                     "class": "MonocleInMemorySpanExporter"}


class TestProviderRegistry(unittest.TestCase):
    def setUp(self):
        self._saved = dict(registry._providers)
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()

    def tearDown(self):
        registry._providers.clear()
        registry._providers.update(self._saved)
        self._env_patcher.stop()

    def test_unknown_name_has_no_provider(self):
        self.assertIsNone(get_provider("no-such-provider"))

    def test_builtin_provider_loads_only_when_asked_for(self):
        target = f"{__name__}:_MemoryProvider"
        with patch.dict(registry._BUILTIN_PROVIDERS, {"builtin-test": target}), \
             patch.object(registry, "import_module", wraps=registry.import_module) as loader:
            self.assertIn("builtin-test", provider_names())
            loader.assert_not_called()
            self.assertIsInstance(get_provider("builtin-test"), _MemoryProvider)
            loader.assert_called_once_with(__name__)

    def test_registered_provider_is_used_as_exporter(self):
        register_provider(_MemoryProvider())
        exporters = get_monocle_exporter(_MemoryProvider.name)
        self.assertEqual([type(e).__name__ for e in exporters], ["MonocleInMemorySpanExporter"])

    def test_entry_point_provider_is_found(self):
        target = f"{__name__}:_MemoryProvider"
        with patch.object(registry, "_entry_point_targets", return_value={"from-entry-point": target}):
            self.assertIn("from-entry-point", provider_names())
            self.assertIsInstance(get_provider("from-entry-point"), _MemoryProvider)

    def test_broken_provider_is_skipped_with_warning(self):
        with patch.object(registry, "_entry_point_targets", return_value={"broken": "no.such.module:X"}):
            with self.assertWarns(UserWarning):
                exporters = get_monocle_exporter("broken")
        # Nothing usable was configured, so the default file exporter is used.
        self.assertEqual([type(e).__name__ for e in exporters], ["FileSpanExporter"])


if __name__ == "__main__":
    unittest.main()
