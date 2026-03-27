from typing import Dict, Any, List
import os
import logging, warnings
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
    "memory": {"module": "monocle_apptrace.exporters.base_exporter", "class": "MonocleInMemorySpanExporter"},
    "console": {"module": "opentelemetry.sdk.trace.export", "class": "ConsoleSpanExporter"},
    "otlp": {"module": "opentelemetry.exporter.otlp.proto.http.trace_exporter", "class": "OTLPSpanExporter"},
    "gcs" : {"module": "monocle_apptrace.exporters.gcp.gcs_exporter", "class": "GCSSpanExporter"}
}


def get_monocle_exporter(exporters_list:str=None) -> List[SpanExporter]:
    # Retrieve the MONOCLE_EXPORTER environment variable and split it into a list
    if exporters_list:
        exporter_names = exporters_list.split(",")
    else:
        exporter_names = os.environ.get("MONOCLE_EXPORTER", "file").split(",")
    exporters = []
    
    # Create task processor for AWS Lambda environment
    task_processor = LambdaExportTaskProcessor() if is_aws_lambda_environment() else None

    for exporter_name in exporter_names:
        exporter_name = exporter_name.strip()
        try:
            exporter_class_path = monocle_exporters[exporter_name]
        except KeyError:
            warnings.warn(f"Unsupported Monocle span exporter '{exporter_name}', skipping.")
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
            warnings.warn(
                f"Unable to initialize Monocle span exporter '{exporter_name}', error: {ex}. Using ConsoleSpanExporter as a fallback.")
            exporters.append(ConsoleSpanExporter())
            continue

    # If no exporters were created, default to FileSpanExporter
    if not exporters:
        logger.debug("No valid Monocle span exporters configured. Defaulting to FileSpanExporter.")
        exporters.append(FileSpanExporter())

    return exporters


def get_filtered_exporter(
    exporter_name: str, 
    filter_config: Dict[str, Any],
    **exporter_kwargs
) -> SpanExporter:
    """
    Create a filtered span exporter.
    
    Wraps a Monocle exporter with a SpanFilter for advanced filtering and projection.
    
    Args:
        exporter_name: Name of the base exporter (e.g., "file", "s3", "blob")
        filter_config: Filter configuration dict with:
            - span_types_to_include: List of span types to include
            - fields_to_include: Dict with "attributes" and "events" to project
            - mode: "include" (default) or "exclude"
        **exporter_kwargs: Additional arguments passed to the base exporter
    
    Returns:
        A FilteredSpanExporter wrapping the base exporter
    
    Example:
        >>> filter_config = {
        ...     "span_types_to_include": ["inference", "inference.framework"],
        ...     "fields_to_include": {
        ...         "attributes": ["entity.1.name", "scope.*"],
        ...         "events": [{"name": "metadata", "attributes": ["completion_tokens"]}]
        ...     }
        ... }
        >>> exporter = get_filtered_exporter("file", filter_config, out_path="./traces")
    """
    from monocle_apptrace.exporters.span_filter import SpanFilter, FilteredSpanExporter
    
    # Get the base exporter
    exporters = get_monocle_exporter(exporter_name)
    if not exporters:
        raise ValueError(f"Could not create exporter '{exporter_name}'")
    
    base_exporter = exporters[0]
    
    # Apply additional kwargs if provided
    if exporter_kwargs:
        # Recreate the exporter with additional kwargs
        try:
            exporter_class_path = monocle_exporters[exporter_name]
            exporter_module = import_module(exporter_class_path["module"])
            exporter_class = getattr(exporter_module, exporter_class_path["class"])
            base_exporter = exporter_class(**exporter_kwargs)
        except Exception as ex:
            logger.warning(f"Could not apply kwargs to exporter: {ex}. Using default.")
    
    # Wrap with filter
    span_filter = SpanFilter(filter_config)
    return FilteredSpanExporter(
        base_exporter=base_exporter,
        span_filter=span_filter
    )
