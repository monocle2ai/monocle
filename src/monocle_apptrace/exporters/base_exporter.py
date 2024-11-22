import time
import random
import logging
from abc import ABC, abstractmethod
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Sequence
import asyncio

logger = logging.getLogger(__name__)

class SpanExporterBase(ABC):
    def __init__(self):
        self.backoff_factor = 2
        self.max_retries = 10
        self.export_queue = []
        self.last_export_time = time.time()

    @abstractmethod
    async def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        pass

    @abstractmethod
    async def force_flush(self, timeout_millis: int = 30000) -> bool:
        pass

    def shutdown(self) -> None:
        pass

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