import json
import pytest
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

class TestAzureBlobNDJSONFormat:

    @pytest.fixture
    def blob_service_client(self):
        connection_string = ""
        client = BlobServiceClient.from_connection_string(connection_string)
        return client

    @pytest.fixture
    def blob_info(self):
        return {
            'container_name': 'sachin',
            'blob_name': 'xx.ndjson'
        }

    def test_blob_ndjson_format(self, blob_service_client, blob_info):
        try:
            blob_client = blob_service_client.get_blob_client(container=blob_info['container_name'], blob=blob_info['blob_name'])
            blob_data = blob_client.download_blob().readall().decode('utf-8')
            lines = blob_data.strip().split("\n")
            for line in lines:
                try:
                    json_obj = json.loads(line)
                    assert isinstance(json_obj, dict), f"Line is not a valid JSON object: {line}"
                except json.JSONDecodeError:
                    raise AssertionError(f"Line is not valid JSON: {line}")

        except ResourceNotFoundError:
            raise AssertionError(f"Blob {blob_info['blob_name']} not found in container {blob_info['container_name']}")

if __name__ == '__main__':
    pytest.main()
