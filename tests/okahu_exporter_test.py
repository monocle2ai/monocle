import unittest
from unittest.mock import patch, MagicMock
from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SpanExportResult
from requests.exceptions import ReadTimeout
from opentelemetry.sdk.trace import ReadableSpan
import json
class TestOkahuSpanExporter(unittest.TestCase):

    @patch.dict('os.environ', {}, clear=True)
    def test_default_to_exception(self):
        """Test that it defaults to exception when no API key is set."""
        exceptionRaised = False
        try:
            exporter = OkahuSpanExporter()
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            exceptionRaised = True
        
        self.assertTrue(exceptionRaised)
                 

    @patch.dict('os.environ', {'OKAHU_API_KEY': 'test-api-key'})
    @patch('monocle_apptrace.exporters.okahu.okahu_exporter.requests.Session')
    def test_okahu_exporter_with_api_key(self, mock_session):
        """Test that OkahuSpanExporter is used when an API key is set."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_post = mock_session_instance.post
        mock_post.return_value.status_code = 200

        mock_span = MagicMock(spec=ReadableSpan)
        mock_span.to_json.return_value = json.dumps({
            "parent_id": "0x123456",
            "context": {
                "trace_id": "0xabcdef",
                "span_id": "0x654321"
            }
        })
        spans = [mock_span]
        exporter = OkahuSpanExporter()
        exporter.export(spans)
        mock_post.assert_called_once()

    @patch.dict('os.environ', {'OKAHU_API_KEY': 'test-api-key'})
    @patch('monocle_apptrace.exporters.okahu.okahu_exporter.requests.Session')
    def test_export_success(self, mock_session):
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.post.return_value.status_code = 200
        exporter = OkahuSpanExporter()
        mock_span = MagicMock()
        mock_span.to_json.return_value = '{"parent_id": null, "context": {"trace_id": "0x123", "span_id": "0x456"}}'

        result = exporter.export([mock_span])
        self.assertEqual(result, SpanExportResult.SUCCESS)

        mock_session_instance.post.assert_called_once_with(
            url=exporter.endpoint,
            data='{"batch": [{"parent_id": "None", "context": {"trace_id": "123", "span_id": "456"}}]}',
            timeout=15
        )

    @patch.dict('os.environ', {'OKAHU_API_KEY': 'test-api-key'})
    @patch('monocle_apptrace.exporters.okahu.okahu_exporter.requests.Session')
    def test_export_failure(self, mock_session):
        """Test exporting spans with an error response from Okahu."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.post.return_value.status_code = 500

        exporter = OkahuSpanExporter()
        mock_span = MagicMock()
        mock_span.to_json.return_value = '{"parent_id": null, "context": {"trace_id": "0x123", "span_id": "0x456"}}'

        result = exporter.export([mock_span])
        self.assertEqual(result, SpanExportResult.FAILURE)

    @patch.dict('os.environ', {'OKAHU_API_KEY': 'test-api-key'})
    @patch('monocle_apptrace.exporters.okahu.okahu_exporter.requests.Session')
    def test_export_timeout(self, mock_session):
        """Test exporting spans with a timeout."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.post.side_effect = ReadTimeout

        exporter = OkahuSpanExporter()
        mock_span = MagicMock()
        mock_span.to_json.return_value = '{"parent_id": null, "context": {"trace_id": "0x123", "span_id": "0x456"}}'

        result = exporter.export([mock_span])
        self.assertEqual(result, SpanExportResult.FAILURE)


if __name__ == '__main__':
    unittest.main()
