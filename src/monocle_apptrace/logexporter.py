import logging
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Sequence
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogExporter(SpanExporter):

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export the spans by logging them using the logger.
        """
        try:
            for span in spans:
                # Log the details of each span
                logger.info(f"Exporting Span:\n{span.to_json()}")

            return SpanExportResult.SUCCESS

        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self):
        logger.info("Shutting down LogExporter.")
        return super().shutdown()

