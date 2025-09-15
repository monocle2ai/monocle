from abc import ABC, abstractmethod
import logging
import os
import queue
import threading
import time
from typing import Callable
import requests
from monocle_apptrace.instrumentation.common.constants import AWS_LAMBDA_ENV_NAME

logger = logging.getLogger(__name__)
LAMBDA_EXTENSION_NAME = "AsyncProcessorMonocle"

class ExportTaskProcessor(ABC):
    
    @abstractmethod
    def start(self):
        return

    @abstractmethod
    def stop(self):
        return

    @abstractmethod
    def queue_task(self, async_task: Callable[[Callable, any], any] = None, args: any = None, is_root_span: bool = False):
        return

class LambdaExportTaskProcessor(ExportTaskProcessor):
    
    def __init__(
        self,
        span_check_interval_seconds: int = 1,
        max_time_allowed_seconds: int = 30):
        # An internal queue used by the handler to notify the extension that it can
        # start processing the async task.
        self.async_tasks_queue = queue.Queue()
        self.span_check_interval = span_check_interval_seconds
        self.max_time_allowed = max_time_allowed_seconds

    def start(self):
        try:
            self._start_async_processor()
        except Exception as e:
            logger.error(f"LambdaExportTaskProcessor| Failed to start. {e}")

    def stop(self):
        return

    def queue_task(self, async_task=None, args=None, is_root_span=False):
        self.async_tasks_queue.put((async_task, args, is_root_span))
    
    def set_sagemaker_model(self, endpoint_name: str, span: dict[str, dict[str, str]]):
        try:
            try:
                import boto3
            except ImportError:
                logger.error("LambdaExportTaskProcessor| Failed to import boto3")
                return
            
            client = boto3.client('sagemaker')
            response = client.describe_endpoint(
                EndpointName=endpoint_name
            )
            endpoint_config_name = response["EndpointConfigName"]
            endpoint_config_response = client.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            model_name = endpoint_config_response["ProductionVariants"][0]["ModelName"]
            model_name_response = client.describe_model(ModelName = model_name)
            model_name_id = ""
            try:
                model_name_id = model_name_response["PrimaryContainer"]["Environment"]["HF_MODEL_ID"]
            except:
                pass
            span["attributes"]["model_name"] = model_name_id
        except Exception as e:
            logger.error(f"LambdaExportTaskProcessor| Failed to get sagemaker model. {e}")

    def update_spans(self, export_args):
        try:
            if 'batch' in export_args:
                for span in export_args["batch"]:
                    try:
                        if len(span["attributes"]["sagemaker_endpoint_name"]) > 0 :
                            self.set_sagemaker_model(endpoint_name=span["attributes"]["sagemaker_endpoint_name"], span=span)
                    except:
                        pass
        except Exception as e:
            logger.error(f"LambdaExportTaskProcessor| Failed to update spans. {e}")

    def _start_async_processor(self):
        # Register internal extension
        logger.debug(f"[{LAMBDA_EXTENSION_NAME}] Registering with Lambda service...")
        response = requests.post(
            url=f"http://{os.environ['AWS_LAMBDA_RUNTIME_API']}/2020-01-01/extension/register",
            json={'events': ['INVOKE']},
            headers={'Lambda-Extension-Name': LAMBDA_EXTENSION_NAME}
        )
        ext_id = response.headers['Lambda-Extension-Identifier']
        logger.debug(f"[{LAMBDA_EXTENSION_NAME}] Registered with ID: {ext_id}")

        def process_tasks():
            while True:
                # Call /next to get notified when there is a new invocation and let
                # Lambda know that we are done processing the previous task.

                logger.debug(f"[{LAMBDA_EXTENSION_NAME}] Waiting for invocation...")
                response = requests.get(
                    url=f"http://{os.environ['AWS_LAMBDA_RUNTIME_API']}/2020-01-01/extension/event/next",
                    headers={'Lambda-Extension-Identifier': ext_id},
                    timeout=None
                )
                root_span_found = False
                # all values in seconds
                total_time_elapsed = 0
                while root_span_found is False and total_time_elapsed < self.max_time_allowed:
                    logger.debug(response.json())
                    # Get next task from internal queue
                    logger.info(f"[{LAMBDA_EXTENSION_NAME}] Async thread running, waiting for task from handler")
                    while self.async_tasks_queue.empty() is False :
                        logger.info(f"[{LAMBDA_EXTENSION_NAME}] Processing task from handler")
                        async_task, arg, is_root_span = self.async_tasks_queue.get()
                        root_span_found = is_root_span
                        # self.update_spans(export_args=arg)

                        if async_task is None:
                            # No task to run this invocation
                            logger.debug(f"[{LAMBDA_EXTENSION_NAME}] Received null task. Ignoring.")
                        else:
                            # Invoke task
                            logger.debug(f"[{LAMBDA_EXTENSION_NAME}] Received async task from handler. Starting task.")
                            async_task(arg)
                    total_time_elapsed+=self.span_check_interval
                    logger.info(f"[{LAMBDA_EXTENSION_NAME}] Waiting for root span. total_time_elapsed: {total_time_elapsed}, root_span_found: {root_span_found}.")
                    time.sleep(self.span_check_interval)
                
                logger.debug(f"[{LAMBDA_EXTENSION_NAME}] Finished processing task. total_time_elapsed: {total_time_elapsed}, root_span_found: {root_span_found}.")

        # Start processing extension events in a separate thread
        threading.Thread(target=process_tasks, daemon=True, name=LAMBDA_EXTENSION_NAME).start() 


def is_aws_lambda_environment():
    return AWS_LAMBDA_ENV_NAME in os.environ