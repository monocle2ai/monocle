import datetime
import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from monocle_apptrace.exporters.file_exporter import FileSpanExporter

logger = logging.getLogger(__name__)

class TestFileSpanExporter(unittest.TestCase):
    def test_file_prefix_default(self):
        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Instantiate the exporter with default prefix
            exporter = FileSpanExporter(out_path=temp_dir)
            
            # Check that the default prefix is used
            self.assertEqual(exporter.file_prefix, "monocle_trace_")

    def test_file_prefix_env_var(self):
        # Set environment variable
        file_prefix = "custom_file_prefix_"
        os.environ['MONOCLE_FILE_PREFIX'] = file_prefix
        
        try:
            # Create a temporary directory for test output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Instantiate the exporter
                exporter = FileSpanExporter(out_path=temp_dir)
                
                # Check that the environment variable prefix is used
                self.assertEqual(exporter.file_prefix, file_prefix)
        finally:
            # Clean up environment variable
            if 'MONOCLE_FILE_PREFIX' in os.environ:
                del os.environ['MONOCLE_FILE_PREFIX']

    def test_file_prefix_constructor_param_overrides_env(self):
        # Set environment variable
        env_prefix = "env_prefix_"
        constructor_prefix = "constructor_prefix_"
        os.environ['MONOCLE_FILE_PREFIX'] = env_prefix
        
        try:
            # Create a temporary directory for test output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Instantiate the exporter with explicit prefix parameter
                exporter = FileSpanExporter(out_path=temp_dir, file_prefix=constructor_prefix)
                
                # Check that the constructor parameter takes precedence
                self.assertEqual(exporter.file_prefix, constructor_prefix)
        finally:
            # Clean up environment variable
            if 'MONOCLE_FILE_PREFIX' in os.environ:
                del os.environ['MONOCLE_FILE_PREFIX']

    def test_file_prefix_in_filename(self):
        # Set environment variable
        file_prefix = "test_custom_"
        os.environ['MONOCLE_FILE_PREFIX'] = file_prefix
        
        try:
            # Create a temporary directory for test output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Mock a ReadableSpan
                mock_span = MagicMock()
                mock_span.context.trace_id = 12345
                mock_span.context.span_id = 67890
                mock_span.parent = None  # Root span
                mock_span.resource.attributes = {"service.name": "test_service"}
                mock_span.to_json.return_value = '{"test": "data"}'
                
                # Instantiate the exporter
                exporter = FileSpanExporter(out_path=temp_dir)
                
                # Process a span to trigger file creation
                with patch('datetime.datetime') as mock_datetime:
                    mock_current_time = datetime.datetime(2024, 12, 10, 10, 0, 0)
                    mock_datetime.now.return_value = mock_current_time
                    mock_datetime.strftime = datetime.datetime.strftime
                    
                    exporter._process_spans([mock_span], is_root_span=True)
                    
                    # Check that the file path contains the custom prefix
                    self.assertIsNotNone(exporter.current_file_path)
                    self.assertIn(file_prefix, exporter.current_file_path)
                    
        finally:
            # Clean up environment variable
            if 'MONOCLE_FILE_PREFIX' in os.environ:
                del os.environ['MONOCLE_FILE_PREFIX']

if __name__ == '__main__':
    unittest.main()