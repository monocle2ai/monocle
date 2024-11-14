import json
import datetime
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, ClientAuthenticationError
from monocle_apptrace.exporters.base_logexporter import BaseLogExporter
from monocle_apptrace.exporters.logging_config import logger
from opentelemetry.sdk.trace.export import SpanExportResult
from typing import Sequence

class AzureBlobLogExporter(BaseLogExporter):
    def __init__(self, container_name = None, connection_string = None, **kwargs):
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = blob_service_client.get_container_client(container_name)

        super().__init__(self.container_client, container_name, **kwargs)

        self.logger = logger  # Use globally configured logger
        self.container_name = container_name

        if not self.__container_exists(self.container_name):
            self.create_container(self.container_name)

    def __container_exists(self, container_name):
        try:
            self.container_client.get_container_properties()
            return True
        except ResourceNotFoundError:
            logger.error(f"Container {container_name} not found (404).")
            return False
        except ClientAuthenticationError:
            logger.error(f"Access to container {container_name} is forbidden (403).")
            raise PermissionError(f"Access to container {container_name} is forbidden.")
        except Exception as e:
            logger.error(f"Unexpected error when checking if container {container_name} exists: {e}")
            raise e

    def create_container(self, container_name):
        """Create the container if it doesn't already exist."""
        try:
            self.container_client.create_container(container_name)
            self.logger.info(f"Container '{container_name}' created successfully.")
        except ResourceExistsError:
            self.logger.info(f"Container '{container_name}' already exists.")
        except Exception as e:
            self.logger.error(f"Error creating container '{container_name}': {e}")
            raise e

    def export_to_storage(self) -> SpanExportResult:
        try:
            self.__upload_logs()
            return SpanExportResult.SUCCESS
        except Exception as e:
            self.logger.error(f"Error uploading to Azure Blob Storage: {e}")
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
        """Upload log file contents from logger.log to Azure Blob Storage."""
        try:
            # Read logs from the local log file
            with open("../tests/logger.log", "r") as log_file:
                log_data = log_file.read()

            # Generate a unique log file name
            current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M.%S")
            log_file_name = f"monocle_logs__{current_time}.log"

            # Upload the log file contents to the blob container
            blob_client = self.container_client.get_blob_client(log_file_name)
            blob_client.upload_blob(log_data, overwrite=True)
            self.logger.info(f"Logs uploaded to Azure Blob Storage as '{log_file_name}'.")
        except Exception as e:
            self.logger.error(f"Error uploading logs to Azure Blob Storage: {e}")
