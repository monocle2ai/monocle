import time
import random
import logging
from abc import ABC, abstractmethod
from azure.core.exceptions import ServiceRequestError, ClientAuthenticationError
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

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Handle retries with exponential backoff."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                return func(*args, **kwargs)
            except ServiceRequestError as e:
                logger.warning(f"Network connectivity error: {e}. Retrying in {self.backoff_factor ** attempt} seconds...")
                sleep_time = self.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(sleep_time)
                attempt += 1
            except ClientAuthenticationError as e:
                logger.error(f"Failed to authenticate: {str(e)}")
                break

        logger.error("Max retries exceeded.")
        raise ServiceRequestError(message="Max retries exceeded.")