import boto3
import datetime
import json
import os
from botocore.exceptions import ClientError
from opentelemetry.sdk.trace.export import SpanExportResult
from monocle_apptrace.exporters.base_logexporter import BaseLogExporter
from monocle_apptrace.exporters.logging_config import logger

class S3LogExporter(BaseLogExporter):
    def __init__(self, bucket_name: str, region_name: str = "us-east-1", **kwargs):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region_name
        )
        super().__init__(self.s3_client, bucket_name, **kwargs)
        self.logger = logger
        self.bucket_name = bucket_name
        self.region_name = region_name

        # Check if bucket exists or create it
        if not self.__bucket_exists(self.bucket_name):
            self.create_bucket()

    def __bucket_exists(self, bucket_name):
        try:
            # Check if the bucket exists by calling head_bucket
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket not found
                logger.error(f"Bucket {bucket_name} does not exist (404).")
                return False
            elif error_code == '403':
                # Permission denied
                logger.error(f"Access to bucket {bucket_name} is forbidden (403).")
                raise PermissionError(f"Access to bucket {bucket_name} is forbidden.")
            elif error_code == '400':
                # Bad request or malformed input
                logger.error(f"Bad request for bucket {bucket_name} (400).")
                raise ValueError(f"Bad request for bucket {bucket_name}.")
            else:
                # Other client errors
                logger.error(f"Unexpected error when accessing bucket {bucket_name}: {e}")
                raise e
        except TypeError as e:
            # Handle TypeError separately
            logger.error(f"Type error while checking bucket existence: {e}")
            raise e

    def create_bucket(self):
        try:
            if self.region_name == "us-east-1":
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.region_name}
                 )
            self.logger.info(f"Bucket {self.bucket_name} created successfully.")
        except ClientError as e:
            self.logger.error(f"Error creating bucket {self.bucket_name}: {e}")
            raise e

    def export_to_storage(self) -> SpanExportResult:
        try:
            # upload logs
            self.__upload_logs()

            return SpanExportResult.SUCCESS
        except Exception as e:
            self.logger.error(f"Error uploading to S3: {e}")
            return SpanExportResult.FAILURE

    def __serialize_spans(self, span_list) -> str:
        valid_json_list = []
        for span in span_list['batch']:
            try:
                valid_json_list.append(json.dumps(span))
            except json.JSONDecodeError as e:
                self.logger.warning(f"Invalid JSON format in span data: {e}")
        return "\n".join(valid_json_list)

    def __upload_logs(self):
        """Upload logs to S3 as a separate file."""
        try:
            # Get the logs from the local logger file
            with open("../tests/logger.log", "r") as log_file:
                log_data = log_file.read()

            current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M.%S")
            log_file_name = f"monocle_logs__{current_time}.log"

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=log_file_name,
                Body=log_data
            )
            self.logger.info(f"Logs uploaded to S3 as {log_file_name}.")
        except Exception as e:
            self.logger.error(f"Error uploading logs to S3: {e}")
