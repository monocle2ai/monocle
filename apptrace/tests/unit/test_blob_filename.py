import os
import unittest
from unittest.mock import MagicMock, patch


class TestAzureBlobSpanExporterPrefix(unittest.TestCase):

    def setUp(self):
        self.original_env = os.environ.copy()
        os.environ.pop('MONOCLE_BLOB_FILE_PREFIX', None)
        os.environ['MONOCLE_BLOB_CONNECTION_STRING'] = (
            "DefaultEndpointsProtocol=https;AccountName=test;"
            "AccountKey=dGVzdA==;EndpointSuffix=core.windows.net"
        )

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch('azure.storage.blob.BlobServiceClient.from_connection_string')
    def test_default_prefix_when_no_env(self, mock_from_conn):
        from monocle_apptrace.exporters.azure.blob_exporter import AzureBlobSpanExporter
        client = MagicMock()
        client.get_container_client.return_value.exists.return_value = True
        mock_from_conn.return_value = client
        exporter = AzureBlobSpanExporter(container_name="test-container")
        self.assertEqual(exporter.file_prefix, "monocle_trace_")

    @patch('azure.storage.blob.BlobServiceClient.from_connection_string')
    def test_env_var_sets_prefix(self, mock_from_conn):
        from monocle_apptrace.exporters.azure.blob_exporter import AzureBlobSpanExporter
        client = MagicMock()
        client.get_container_client.return_value.exists.return_value = True
        mock_from_conn.return_value = client
        os.environ['MONOCLE_BLOB_FILE_PREFIX'] = 'mad_trace_'
        exporter = AzureBlobSpanExporter(container_name="test-container")
        self.assertEqual(exporter.file_prefix, 'mad_trace_')


if __name__ == '__main__':
    unittest.main()
