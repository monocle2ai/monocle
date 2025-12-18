import datetime
import logging
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from monocle_apptrace.exporters.aws.s3_exporter import S3SpanExporter
from monocle_apptrace.exporters.base_exporter import format_trace_id_without_0x

logger = logging.getLogger(__name__)

class TestS3SpanExporter(unittest.TestCase):
    @patch('boto3.client')
    def test_file_prefix_in_file_name(self, mock_boto_client):
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        # Instantiate the exporter with a custom prefix
        exporter = S3SpanExporter(bucket_name="test-bucket", region_name="us-east-1")
        file_prefix = "monocle_trace_"
        # Mock current time for consistency
        mock_current_time = datetime.datetime(2024, 12, 10, 10, 0, 0)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_current_time
            mock_datetime.strftime = datetime.datetime.strftime

            # Call the private method to upload data (we are testing the file naming logic)
            test_span_data = "{\"trace_id\": \"123\"}"
            test_trace_id = 0x123456789abcdef0123456789abcdef0
            exporter._S3SpanExporter__upload_to_s3_with_trace_id(test_span_data, test_trace_id)

            # Generate expected file name
            expected_file_name = f"{file_prefix}{mock_current_time.strftime(exporter.time_format)}_{format_trace_id_without_0x(test_trace_id)}.ndjson"

            # Verify the S3 client was called with the correct file name
            mock_s3_client.put_object.assert_called_once_with(
                Bucket="test-bucket",
                Key=expected_file_name,
                Body=test_span_data
            )
    
    @patch('boto3.client')
    def test_file_prefix_in_file_name_env(self, mock_boto_client):
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        file_prefix = "test_prefix_2"
        os.environ['MONOCLE_S3_KEY_PREFIX'] = file_prefix
        # Instantiate the exporter with a custom prefix
        exporter = S3SpanExporter(bucket_name="test-bucket", region_name="us-east-1")
        
        # Mock current time for consistency
        mock_current_time = datetime.datetime(2024, 12, 10, 10, 0, 0)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_current_time
            mock_datetime.strftime = datetime.datetime.strftime

            # Call the private method to upload data (we are testing the file naming logic)
            test_span_data = "{\"trace_id\": \"123\"}"
            test_trace_id = 0x123456789abcdef0123456789abcdef0
            exporter._S3SpanExporter__upload_to_s3_with_trace_id(test_span_data, test_trace_id)

            # Generate expected file name
            expected_file_name = f"{file_prefix}{mock_current_time.strftime(exporter.time_format)}_{format_trace_id_without_0x(test_trace_id)}.ndjson"

            # Verify the S3 client was called with the correct file name
            mock_s3_client.put_object.assert_called_once_with(
                Bucket="test-bucket",
                Key=expected_file_name,
                Body=test_span_data
            )

if __name__ == '__main__':
    unittest.main()
