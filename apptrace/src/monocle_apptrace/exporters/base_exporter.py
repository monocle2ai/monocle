import time, os
import json
import asyncio
import random
import logging
from abc import ABC, abstractmethod
from functools import wraps
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SDK_VERSION
from typing import Sequence

logger = logging.getLogger(__name__)

class SpanExporterBase(ABC):
    def __init__(self, export_monocle_only: bool = True):
        self.backoff_factor = 2
        self.max_retries = 10
        self.export_queue = []
        self.last_export_time = time.time()
        self.export_monocle_only = export_monocle_only or os.environ.get("MONOCLE_EXPORTS_ONLY", True)

    def __init_subclass__(cls, **kwargs):
        """Route every subclass's export() through obfuscate_spans() first.

        Wrapping here rather than in each exporter means sensitive data.input /
        data.output payloads are scrubbed for all exporters, including any added
        later, with no per-exporter code. Obfuscation is idempotent, so spans
        already scrubbed by the span-processor hook pass straight through.
        """
        super().__init_subclass__(**kwargs)
        export = cls.__dict__.get("export")
        if export is None or getattr(export, "_monocle_obfuscates", False):
            return

        @wraps(export)
        def obfuscating_export(self, spans, *args, **kwargs):
            return export(self, self.obfuscate_spans(spans), *args, **kwargs)

        obfuscating_export._monocle_obfuscates = True
        cls.export = obfuscating_export

    @abstractmethod
    async def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        pass

    @abstractmethod
    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        pass

    def shutdown(self) -> None:
        pass

    def skip_export(self, span:ReadableSpan) -> bool:
        if self.export_monocle_only and (not span.attributes.get(MONOCLE_SDK_VERSION)):
            return True
        return False

    def obfuscate_spans(self, spans: Sequence[ReadableSpan]) -> Sequence[ReadableSpan]:
        """Scrub sensitive data from the spans' data.input/data.output payloads.

        Called for every exporter via ``__init_subclass__``. Idempotent, so
        calling it on already-obfuscated spans is a no-op. Override to change
        which obfuscators a specific exporter applies.
        """
        from monocle_apptrace.exporters.span_obfuscator import (
            get_span_obfuscators,
            obfuscate_spans as _obfuscate_spans,
        )
        return _obfuscate_spans(spans, get_span_obfuscators())

    @staticmethod
    def _is_running_in_event_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    @staticmethod
    def retry_with_backoff(retries=3, backoff_in_seconds=1, max_backoff_in_seconds=32, exceptions=(Exception,)):
        def decorator(func):
            def wrapper(*args, **kwargs):
                attempt = 0
                while attempt < retries:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        attempt += 1
                        sleep_time = min(max_backoff_in_seconds, backoff_in_seconds * (2 ** (attempt - 1)))
                        sleep_time = sleep_time * (1 + random.uniform(-0.1, 0.1))  # Add jitter
                        logger.warning(f"Network connectivity error, Attempt {attempt} failed: {e}. Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                raise Exception(f"Failed after {retries} attempts")

            return wrapper

        return decorator


class MonocleInMemorySpanExporter(InMemorySpanExporter, SpanExporterBase):
    """In-memory span exporter that keeps only Monocle-instrumented spans."""

    def __init__(self, export_monocle_only: bool = True):
        InMemorySpanExporter.__init__(self)
        SpanExporterBase.__init__(self, export_monocle_only=export_monocle_only)

    def export(self, spans):
        filtered_spans = [span for span in spans if not self.skip_export(span)]
        if not filtered_spans:
            return SpanExportResult.SUCCESS
        return super().export(filtered_spans)
    

def format_trace_id_without_0x(trace_id: int) -> str:
    """Format trace_id as 32-character lowercase hex string without 0x prefix."""
    return f"{trace_id:032x}"

def format_span_id_without_0x(span_id: int) -> str:
    """Format span_id as 16-character lowercase hex string without 0x prefix."""
    return f"{span_id:016x}"

def serialize_span(span) -> dict:
    """Serialize a ReadableSpan to a dict using OTLP JSON field names.

    OTel's to_json() uses 'description' for the status message; OTLP JSON
    (and the Monocle backend) expects 'message'.  This function normalizes
    the key so all exporters produce consistent output.
    """
    obj = json.loads(span.to_json())
    status = obj.get("status", {})
    if "description" in status:
        status["message"] = status.pop("description")
    return obj
