import datetime
import logging
import os
import unittest
import warnings
from unittest.mock import MagicMock, patch

from monocle_apptrace.exporters.aws.s3_exporter import S3SpanExporter
from monocle_apptrace.exporters.base_exporter import format_trace_id_without_0x

logger = logging.getLogger(__name__)


class TestS3SpanExporter(unittest.TestCase):

    def setUp(self):
        self.original_env = os.environ.copy()
        for v in ('MONOCLE_S3_FILE_PREFIX', 'MONOCLE_S3_KEY_PREFIX', 'MONOCLE_S3_KEY_PREFIX_CURRENT'):
            os.environ.pop(v, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def _expected_name(self, prefix, exporter, trace_id, now):
        return f"{prefix}{now.strftime(exporter.time_format)}_{format_trace_id_without_0x(trace_id)}.ndjson"

    @patch('boto3.client')
    def test_default_prefix_when_no_env(self, mock_boto_client):
        mock_boto_client.return_value = MagicMock()
        exporter = S3SpanExporter(bucket_name="test-bucket", region_name="us-east-1")
        self.assertEqual(exporter.file_prefix, "monocle_trace_")

    @patch('boto3.client')
    def test_new_env_var_sets_prefix(self, mock_boto_client):
        mock_boto_client.return_value = MagicMock()
        os.environ['MONOCLE_S3_FILE_PREFIX'] = 'mad_trace_'
        exporter = S3SpanExporter(bucket_name="test-bucket", region_name="us-east-1")
        self.assertEqual(exporter.file_prefix, 'mad_trace_')

    @patch('boto3.client')
    def test_legacy_env_var_still_works_with_deprecation_warning(self, mock_boto_client):
        mock_boto_client.return_value = MagicMock()
        os.environ['MONOCLE_S3_KEY_PREFIX'] = 'legacy_'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exporter = S3SpanExporter(bucket_name="test-bucket", region_name="us-east-1")
        self.assertEqual(exporter.file_prefix, 'legacy_')
        deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertTrue(deprecation, "expected a DeprecationWarning for MONOCLE_S3_KEY_PREFIX")
        self.assertIn('MONOCLE_S3_FILE_PREFIX', str(deprecation[0].message))

    @patch('boto3.client')
    def test_new_env_var_takes_precedence_over_legacy(self, mock_boto_client):
        mock_boto_client.return_value = MagicMock()
        os.environ['MONOCLE_S3_FILE_PREFIX'] = 'new_'
        os.environ['MONOCLE_S3_KEY_PREFIX'] = 'old_'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exporter = S3SpanExporter(bucket_name="test-bucket", region_name="us-east-1")
        self.assertEqual(exporter.file_prefix, 'new_')
        # When the new var is set, the legacy one is silently ignored — no warning.
        deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertFalse(deprecation, "should not warn when new var is present")

    @patch('boto3.client')
    def test_file_prefix_in_file_name(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        exporter = S3SpanExporter(bucket_name="test-bucket", region_name="us-east-1")
        now = datetime.datetime(2024, 12, 10, 10, 0, 0)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.strftime = datetime.datetime.strftime
            test_span_data = "{\"trace_id\": \"123\"}"
            test_trace_id = 0x123456789abcdef0123456789abcdef0
            exporter._S3SpanExporter__upload_to_s3_with_trace_id(test_span_data, test_trace_id)
            mock_s3_client.put_object.assert_called_once_with(
                Bucket="test-bucket",
                Key=self._expected_name("monocle_trace_", exporter, test_trace_id, now),
                Body=test_span_data,
            )

    @patch('boto3.client')
    def test_file_prefix_in_file_name_new_env(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        os.environ['MONOCLE_S3_FILE_PREFIX'] = 'mad_trace_'
        exporter = S3SpanExporter(bucket_name="test-bucket", region_name="us-east-1")
        now = datetime.datetime(2024, 12, 10, 10, 0, 0)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.strftime = datetime.datetime.strftime
            test_span_data = "{\"trace_id\": \"123\"}"
            test_trace_id = 0x123456789abcdef0123456789abcdef0
            exporter._S3SpanExporter__upload_to_s3_with_trace_id(test_span_data, test_trace_id)
            mock_s3_client.put_object.assert_called_once_with(
                Bucket="test-bucket",
                Key=self._expected_name("mad_trace_", exporter, test_trace_id, now),
                Body=test_span_data,
            )


if __name__ == '__main__':
    unittest.main()
