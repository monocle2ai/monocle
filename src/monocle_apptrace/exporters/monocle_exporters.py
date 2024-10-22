from typing import Dict, Any
import os, warnings
from importlib import import_module
from opentelemetry.sdk.trace.export import SpanExporter, ConsoleSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter

monocle_exporters:Dict[str, Any] = {
    "s3": {"module": "monocle_apptrace.exporters.aws.s3_exporter", "class": "S3SpanExporter"},
    "blob": {"module":"monocle_apptrace.exporters.azure.blob_exporter", "class": "AzureBlobSpanExporter"},
    "okahu": {"module":"monocle_apptrace.exporters.okahu.okahu_exporter", "class": "OkahuSpanExporter"},
    "file": {"module":"monocle_apptrace.exporters.file_exporter", "class": "FileSpanExporter"}
}

def get_monocle_exporter() -> SpanExporter:
    exporter_name = os.environ.get("MONOCLE_EXPORTER", "file")
    try:
        exporter_class_path  = monocle_exporters[exporter_name]
    except Exception as ex:
        warnings.warn(f"Unsupported Monocle span exporter setting {exporter_name}, using default FileSpanExporter.")
        return FileSpanExporter()
    try:
        exporter_module = import_module(exporter_class_path.get("module"))
        exporter_class = getattr(exporter_module, exporter_class_path.get("class"))
        return exporter_class()
    except Exception as ex:
        warnings.warn(f"Unable to set Monocle span exporter to {exporter_name}, error {ex}. Using ConsoleSpanExporter")
        return ConsoleSpanExporter()