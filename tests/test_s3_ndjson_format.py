import unittest
import json
import boto3
from botocore.exceptions import NoCredentialsError


class TestS3NDJSONFormat(unittest.TestCase):

    def setUp(self):
        self.s3_client = boto3.client('s3', aws_access_key_id='',
                                      aws_secret_access_key='', region_name='us-east-1')
        self.bucket_name = 'sachin-dev'
        self.s3_file_key = 'xx.ndjson'

    def test_s3_ndjson_format(self):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.s3_file_key)
            file_content = response['Body'].read().decode('utf-8')

            lines = file_content.strip().split("\n")
            for line in lines:
                try:
                    json_obj = json.loads(line)
                    self.assertIsInstance(json_obj, dict, "Line is not a valid JSON object")
                except json.JSONDecodeError:
                    self.fail(f"Line is not valid JSON: {line}")

        except NoCredentialsError:
            self.fail("AWS credentials not available")

    def tearDown(self):
        self.s3_client = None


if __name__ == '__main__':
    unittest.main()
