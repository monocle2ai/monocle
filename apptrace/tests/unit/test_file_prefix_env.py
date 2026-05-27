import os
import unittest

from monocle_apptrace.exporters.file_exporter import (
    DEFAULT_FILE_PREFIX,
    FileSpanExporter,
)


class TestFileSpanExporterPrefix(unittest.TestCase):

    def setUp(self):
        self.original_env = os.environ.copy()
        os.environ.pop('MONOCLE_FILE_PREFIX', None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_default_prefix_when_no_arg_and_no_env(self):
        exporter = FileSpanExporter()
        self.assertEqual(exporter.file_prefix, DEFAULT_FILE_PREFIX)

    def test_env_var_sets_prefix(self):
        os.environ['MONOCLE_FILE_PREFIX'] = 'mad_trace_'
        exporter = FileSpanExporter()
        self.assertEqual(exporter.file_prefix, 'mad_trace_')

    def test_explicit_arg_overrides_env(self):
        os.environ['MONOCLE_FILE_PREFIX'] = 'from_env_'
        exporter = FileSpanExporter(file_prefix='from_arg_')
        self.assertEqual(exporter.file_prefix, 'from_arg_')

    def test_explicit_arg_used_when_no_env(self):
        exporter = FileSpanExporter(file_prefix='only_arg_')
        self.assertEqual(exporter.file_prefix, 'only_arg_')


if __name__ == '__main__':
    unittest.main()
