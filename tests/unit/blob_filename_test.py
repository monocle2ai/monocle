import datetime
import logging
import os
import unittest
from unittest.mock import MagicMock, patch

from monocle_apptrace.exporters.azure.blob_exporter import AzureBlobSpanExporter

logger = logging.getLogger(__name__)

class TestAzureBlobSpanExporter(unittest.TestCase):
    @patch('azure.storage.blob.BlobServiceClient.from_connection_string')
    def test_file_prefix_default(self, mock_blob_service_client):
        # Mock blob service client
        mock_client = MagicMock()
        mock_blob_service_client.return_value = mock_client
        mock_container_client = MagicMock()
        mock_client.get_container_client.return_value = mock_container_client
        mock_container_client.get_container_properties.return_value = {}
        
        # Instantiate the exporter
        exporter = AzureBlobSpanExporter(
            connection_string="DefaultEndpointsProtocol=https;AccountName=test", 
            container_name="test-container"
        )
        
        # Mock current time for consistency
        mock_current_time = datetime.datetime(2024, 12, 10, 10, 0, 0)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_current_time
            mock_datetime.strftime = datetime.datetime.strftime

            # Mock blob client
            mock_blob_client = MagicMock()
            mock_client.get_blob_client.return_value = mock_blob_client

            # Call the private method to upload data
            test_span_data = "{\"trace_id\": \"123\"}"
            exporter._AzureBlobSpanExporter__upload_to_blob(test_span_data)

            # Generate expected file name with default prefix
            expected_file_name = f"monocle_trace_{mock_current_time.strftime(exporter.time_format)}.ndjson"

            # Verify the blob client was called with the correct file name
            mock_client.get_blob_client.assert_called_once_with(
                container="test-container",
                blob=expected_file_name
            )
            mock_blob_client.upload_blob.assert_called_once_with(test_span_data, overwrite=True)

    @patch('azure.storage.blob.BlobServiceClient.from_connection_string')
    def test_file_prefix_env_var(self, mock_blob_service_client):
        # Mock blob service client
        mock_client = MagicMock()
        mock_blob_service_client.return_value = mock_client
        mock_container_client = MagicMock()
        mock_client.get_container_client.return_value = mock_container_client
        mock_container_client.get_container_properties.return_value = {}
        
        # Set environment variable
        file_prefix = "custom_blob_prefix_"
        os.environ['MONOCLE_BLOB_FILE_PREFIX'] = file_prefix
        
        # Instantiate the exporter
        exporter = AzureBlobSpanExporter(
            connection_string="DefaultEndpointsProtocol=https;AccountName=test", 
            container_name="test-container"
        )
        
        # Mock current time for consistency
        mock_current_time = datetime.datetime(2024, 12, 10, 10, 0, 0)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_current_time
            mock_datetime.strftime = datetime.datetime.strftime

            # Mock blob client
            mock_blob_client = MagicMock()
            mock_client.get_blob_client.return_value = mock_blob_client

            # Call the private method to upload data
            test_span_data = "{\"trace_id\": \"123\"}"
            exporter._AzureBlobSpanExporter__upload_to_blob(test_span_data)

            # Generate expected file name with custom prefix
            expected_file_name = f"{file_prefix}{mock_current_time.strftime(exporter.time_format)}.ndjson"

            # Verify the blob client was called with the correct file name
            mock_client.get_blob_client.assert_called_once_with(
                container="test-container",
                blob=expected_file_name
            )
            mock_blob_client.upload_blob.assert_called_once_with(test_span_data, overwrite=True)
        
        # Clean up environment variable
        if 'MONOCLE_BLOB_FILE_PREFIX' in os.environ:
            del os.environ['MONOCLE_BLOB_FILE_PREFIX']

if __name__ == '__main__':
    unittest.main()