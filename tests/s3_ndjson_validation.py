import json
import boto3
from botocore.exceptions import NoCredentialsError
import pytest


class TestS3NDJSONFormat:

    @pytest.fixture
    def s3_client(self):
        client = boto3.client(
            's3',
            aws_access_key_id='',
            aws_secret_access_key='',
            region_name='us-east-1'
        )
        return client

    @pytest.fixture
    def bucket_info(self):
        return {
            'bucket_name': 'sachin-dev',
            's3_file_key': 'xx.ndjson'
        }

    def test_s3_ndjson_format(self, s3_client, bucket_info):
        try:
            response = s3_client.get_object(Bucket=bucket_info['bucket_name'], Key=bucket_info['s3_file_key'])
            file_content = response['Body'].read().decode('utf-8')

            lines = file_content.strip().split("\n")
            for line in lines:
                try:
                    json_obj = json.loads(line)
                    assert isinstance(json_obj, dict), f"Line is not a valid JSON object: {line}"
                except json.JSONDecodeError:
                    raise AssertionError(f"Line is not valid JSON: {line}")

        except NoCredentialsError:
            raise AssertionError("AWS credentials not available")
        except Exception as e:
            raise AssertionError(f"Test failed with error: {e}")


if __name__ == '__main__':
    pytest.main()
