from typing import Dict, Any, List
import os
import logging
from importlib import import_module
from opentelemetry.sdk.trace.export import SpanExporter, ConsoleSpanExporter
from monocle_apptrace.exporters.exporter_processor import LambdaExportTaskProcessor, is_aws_lambda_environment
from monocle_apptrace.exporters.file_exporter import FileSpanExporter

logger = logging.getLogger(__name__)

monocle_exporters: Dict[str, Any] = {
    "s3": {"module": "monocle_apptrace.exporters.aws.s3_exporter", "class": "S3SpanExporter"},
    "blob": {"module": "monocle_apptrace.exporters.azure.blob_exporter", "class": "AzureBlobSpanExporter"},
    "okahu": {"module": "monocle_apptrace.exporters.okahu.okahu_exporter", "class": "OkahuSpanExporter"},
    "file": {"module": "monocle_apptrace.exporters.file_exporter", "class": "FileSpanExporter"},
    "memory": {"module": "opentelemetry.sdk.trace.export.in_memory_span_exporter", "class": "InMemorySpanExporter"},
    "console": {"module": "opentelemetry.sdk.trace.export", "class": "ConsoleSpanExporter"}
}


def get_monocle_exporter() -> List[SpanExporter]:
    # Retrieve the MONOCLE_EXPORTER environment variable and split it into a list
    exporter_names = os.environ.get("MONOCLE_EXPORTER", "file").split(",")
    exporters = []
    
    # Create task processor for AWS Lambda environment
    task_processor = LambdaExportTaskProcessor() if is_aws_lambda_environment() else None

    for exporter_name in exporter_names:
        exporter_name = exporter_name.strip()
        try:
            exporter_class_path = monocle_exporters[exporter_name]
        except KeyError:
            logger.debug(f"Unsupported Monocle span exporter '{exporter_name}', skipping.")
            continue
        try:
            exporter_module = import_module(exporter_class_path["module"])
            exporter_class = getattr(exporter_module, exporter_class_path["class"])
            # Pass task_processor to all exporters when in AWS Lambda environment
            if task_processor is not None and exporter_module.__name__.startswith("monocle_apptrace"):
                exporters.append(exporter_class(task_processor=task_processor))
            else:
                exporters.append(exporter_class())
        except Exception as ex:
            logger.debug(
                f"Unable to initialize Monocle span exporter '{exporter_name}', error: {ex}. Using ConsoleSpanExporter as a fallback.")
            exporters.append(ConsoleSpanExporter())
            continue

    # If no exporters were created, default to FileSpanExporter
    if not exporters:
        logger.debug("No valid Monocle span exporters configured. Defaulting to FileSpanExporter.")
        exporters.append(FileSpanExporter())

    return exporters
