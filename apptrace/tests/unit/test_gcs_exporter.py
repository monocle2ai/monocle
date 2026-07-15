import datetime
import logging
import os
import unittest
from unittest.mock import MagicMock, patch, Mock

from monocle_apptrace.exporters.gcp.gcs_exporter import GCSSpanExporter
from monocle_apptrace.exporters.base_exporter import format_trace_id_without_0x
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

logger = logging.getLogger(__name__)


class TestGCSSpanExporter(unittest.TestCase):

    def setUp(self):
        self.original_env = os.environ.copy()
        os.environ.pop('MONOCLE_GCS_BUCKET_NAME', None)
        os.environ.pop('MONOCLE_GCS_PROJECT_ID', None)
        os.environ.pop('MONOCLE_GCS_LOCATION', None)
        os.environ.pop('MONOCLE_GCS_KEY_PREFIX', None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch('google.cloud.storage.Client')
    def test_file_prefix_in_file_name(self, mock_storage_client):
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = True
        
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        exporter = GCSSpanExporter(bucket_name="test-bucket", project_id="test-project")

        mock_current_time = datetime.datetime(2026, 1, 30, 10, 0, 0)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_current_time
            mock_datetime.strftime = datetime.datetime.strftime

            test_span_data = '{"trace_id": "123"}\n'
            test_trace_id = 0x123456789abcdef0123456789abcdef0
            exporter._GCSSpanExporter__upload_to_gcs_with_trace_id(test_span_data, test_trace_id)

            expected_file_name = f"monocle_trace_{mock_current_time.strftime(exporter.time_format)}_{format_trace_id_without_0x(test_trace_id)}.ndjson"

            mock_bucket.blob.assert_called_once_with(expected_file_name)
            mock_blob.upload_from_string.assert_called_once_with(
                data=test_span_data,
                content_type='application/x-ndjson'
            )

    @patch('google.cloud.storage.Client')
    def test_file_prefix_from_env_variable(self, mock_storage_client):
        # Mock GCS client and bucket
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = True
        
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        custom_prefix = "custom_prefix_"
        os.environ['MONOCLE_GCS_KEY_PREFIX'] = custom_prefix
        exporter = GCSSpanExporter(bucket_name="test-bucket", project_id="test-project")

        mock_current_time = datetime.datetime(2026, 1, 30, 10, 0, 0)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_current_time
            mock_datetime.strftime = datetime.datetime.strftime

            test_span_data = '{"trace_id": "123"}\n'
            test_trace_id = 0xabcdefabcdefabcdefabcdefabcdefab
            exporter._GCSSpanExporter__upload_to_gcs_with_trace_id(test_span_data, test_trace_id)

            expected_file_name = f"{custom_prefix}{mock_current_time.strftime(exporter.time_format)}_{format_trace_id_without_0x(test_trace_id)}.ndjson"

            mock_bucket.blob.assert_called_once_with(expected_file_name)

    @patch('google.cloud.storage.Client')
    def test_bucket_name_from_env_variable(self, mock_storage_client):

        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = True

        os.environ['MONOCLE_GCS_BUCKET_NAME'] = 'env-test-bucket'

        exporter = GCSSpanExporter(project_id="test-project")

        self.assertEqual(exporter.bucket_name, 'env-test-bucket')

    @patch('google.cloud.storage.Client')
    def test_missing_bucket_name_raises_error(self, mock_storage_client):
        with self.assertRaises(ValueError) as context:
            GCSSpanExporter(project_id="test-project")
        
        self.assertIn("GCS bucket name is not provided", str(context.exception))

    @patch('google.cloud.storage.Client')
    def test_bucket_creation_when_not_exists(self, mock_storage_client):
        # Mock GCS client
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"

        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = False

        mock_client_instance.create_bucket.return_value = mock_bucket
        exporter = GCSSpanExporter(
            bucket_name="new-test-bucket",
            project_id="test-project",
            location="US"
        )

        mock_client_instance.create_bucket.assert_called_once_with(
            bucket_or_name="new-test-bucket",
            location="US"
        )

    @patch('google.cloud.storage.Client')
    def test_project_id_auto_detection(self, mock_storage_client):
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "auto-detected-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = True

        exporter = GCSSpanExporter(bucket_name="test-bucket")
        self.assertEqual(exporter.project_id, "auto-detected-project")

    @patch('google.cloud.storage.Client')
    def test_trace_buffering(self, mock_storage_client):
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = True

        exporter = GCSSpanExporter(bucket_name="test-bucket", project_id="test-project")

        mock_span1 = MagicMock(spec=ReadableSpan)
        mock_span2 = MagicMock(spec=ReadableSpan)
        
        trace_id = 0x123456789abcdef0123456789abcdef0

        exporter._add_spans_to_trace(trace_id, [mock_span1], has_root=False)
        self.assertIn(trace_id, exporter.trace_spans)
        self.assertEqual(len(exporter.trace_spans[trace_id][0]), 1)
        exporter._add_spans_to_trace(trace_id, [mock_span2], has_root=True)

        self.assertEqual(len(exporter.trace_spans[trace_id][0]), 2)
        self.assertTrue(exporter.trace_spans[trace_id][2])  # has_root flag

    @patch('google.cloud.storage.Client')
    def test_cleanup_expired_traces(self, mock_storage_client):
        """Test that expired traces are cleaned up and uploaded."""
        # Mock GCS client
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = True
        
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        exporter = GCSSpanExporter(bucket_name="test-bucket", project_id="test-project")
        
        # Create mock span with to_json method
        mock_span = MagicMock(spec=ReadableSpan)
        mock_span.to_json.return_value = '{"test": "span"}'
        mock_span.context.span_id = 123
        
        trace_id = 0x123456789abcdef0123456789abcdef0

        old_time = datetime.datetime.now() - datetime.timedelta(seconds=61)
        exporter.trace_spans[trace_id] = ([mock_span], old_time, False)
        exporter._cleanup_expired_traces()
        self.assertNotIn(trace_id, exporter.trace_spans)
        mock_bucket.blob.assert_called()

    @patch('google.cloud.storage.Client')
    def test_span_serialization(self, mock_storage_client):
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = True

        exporter = GCSSpanExporter(bucket_name="test-bucket", project_id="test-project")

        mock_span1 = MagicMock(spec=ReadableSpan)
        mock_span1.to_json.return_value = '{"span": 1}'
        mock_span1.context.span_id = 1
        
        mock_span2 = MagicMock(spec=ReadableSpan)
        mock_span2.to_json.return_value = '{"span": 2}'
        mock_span2.context.span_id = 2
        result = exporter._GCSSpanExporter__serialize_spans([mock_span1, mock_span2])
        expected = '{"span": 1}\n{"span": 2}\n'
        self.assertEqual(result, expected)

    @patch('google.cloud.storage.Client')
    def test_export_returns_success(self, mock_storage_client):
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = True

        exporter = GCSSpanExporter(bucket_name="test-bucket", project_id="test-project")

        with patch.object(exporter, '_export_async'):
            result = exporter.export([])
            self.assertEqual(result, SpanExportResult.SUCCESS)

    @patch('google.cloud.storage.Client')
    def test_location_configuration(self, mock_storage_client):
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = False
        
        mock_client_instance.create_bucket.return_value = mock_bucket
        exporter = GCSSpanExporter(
            bucket_name="test-bucket",
            project_id="test-project",
            location="us-central1"
        )

        mock_client_instance.create_bucket.assert_called_once_with(
            bucket_or_name="test-bucket",
            location="us-central1"
        )

    @patch('google.cloud.storage.Client')
    def test_location_from_env_variable(self, mock_storage_client):
        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.project = "test-project"
        
        mock_bucket = MagicMock()
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.exists.return_value = False
        
        mock_client_instance.create_bucket.return_value = mock_bucket
        os.environ['MONOCLE_GCS_LOCATION'] = 'EU'
        
        exporter = GCSSpanExporter(
            bucket_name="test-bucket",
            project_id="test-project"
        )

        mock_client_instance.create_bucket.assert_called_once_with(
            bucket_or_name="test-bucket",
            location="EU"
        )


if __name__ == '__main__':
    unittest.main()
