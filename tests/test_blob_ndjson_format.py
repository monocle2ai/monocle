import unittest
import json
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError


class TestAzureBlobNDJSONFormat(unittest.TestCase):

    def setUp(self):
        connection_string = ''
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = 'sachin'
        self.blob_name = 'xx.ndjson'

    def test_blob_ndjson_format(self):
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=self.blob_name)
            blob_data = blob_client.download_blob().readall().decode('utf-8')
            lines = blob_data.strip().split("\n")
            for line in lines:
                try:
                    json_obj = json.loads(line)
                    self.assertIsInstance(json_obj, dict, "Line is not a valid JSON object")
                except json.JSONDecodeError:
                    self.fail(f"Line is not valid JSON: {line}")

        except ResourceNotFoundError:
            self.fail(f"Blob {self.blob_name} not found in container {self.container_name}")

    def tearDown(self):
        self.blob_service_client = None


if __name__ == '__main__':
    unittest.main()
