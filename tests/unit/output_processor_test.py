
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
import unittest
from unittest.mock import Mock
import logging
import os

# Initialize the logger for testing
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TestProcessSpan(unittest.TestCase):

    def setUp(self):
        # Mock the span and instance for tests
        self.mock_span = Mock()
        self.mock_instance = Mock()
        self.mock_args = {}
        self.mock_kwargs = {}
        self.return_value = ""
        self.handler = SpanHandler()
        self.wrapped = Mock()

    def test_valid_output_processor(self):
        """Test case for valid output processor with type and attributes."""
        to_wrap = {
            "output_processor": {
                "type": "inference",
                "attributes": [
                    [
                        {
                            "attribute": "provider_name",
                            "accessor": lambda args: 'example.com'
                        },
                        {
                            "attribute": "inference_endpoint",
                            "accessor": lambda args: 'https://example.com/'
                        }
                    ]
                ]
            }
        }

        self.handler.hydrate_span(to_wrap=to_wrap, wrapped=self.wrapped, span=self.mock_span, instance=self.mock_instance, args=self.mock_args, kwargs=self.mock_kwargs,
                                  result=self.return_value)
        self.mock_span.set_attribute.assert_any_call("span.type", "inference")
        self.mock_span.set_attribute.assert_any_call("entity.count", 1)
        self.mock_span.set_attribute.assert_any_call("entity.1.provider_name", "example.com")
        self.mock_span.set_attribute.assert_any_call("entity.1.inference_endpoint", "https://example.com/")


    def test_output_processor_missing_span_type(self):
        """Test case when type is missing from output processor."""
        to_wrap = {
                "output_processor" : {
                "attributes": [
                    [
                        {
                            "attribute": "provider_name",
                            "accessor": lambda args: 'example.com'
                        },
                        {
                            "attribute": "inference_endpoint",
                            "accessor": lambda args: 'https://example.com/'
                        }
                    ]
                ]
            }
         }
        self.handler.hydrate_span(to_wrap=to_wrap, wrapped=self.wrapped, span=self.mock_span,
                                  instance=self.mock_instance, args=self.mock_args, kwargs=self.mock_kwargs,
                                  result=self.return_value)

        self.mock_span.set_attribute.assert_any_call("entity.count", 1)
        self.mock_span.set_attribute.assert_any_call("entity.1.provider_name", "example.com")


    def test_output_processor_missing_attributes(self):
        """Test case when attributes are missing from output processor."""
        to_wrap ={
                    "output_processor" :
                    {
                           "type": "inference",
                            "attributes":[]
                    }
                }
        self.handler.hydrate_span(to_wrap=to_wrap, wrapped=self.wrapped, span=self.mock_span,
                                  instance=self.mock_instance, args=self.mock_args, kwargs=self.mock_kwargs,
                                  result=self.return_value)

        self.mock_span.set_attribute.assert_any_call("span.type", "inference")

    def test_empty_output_processor(self):
        """Test case for an empty output processor."""
        to_wrap={
            "output_processor":{}
        }
        self.handler.hydrate_span(to_wrap=to_wrap, wrapped=self.wrapped, span=self.mock_span,
                                  instance=self.mock_instance, args=self.mock_args, kwargs=self.mock_kwargs,
                                  result=self.return_value)

        # Log warning expected for incorrect format
        with self.assertLogs(level='WARNING') as log:
            self.handler.hydrate_span(to_wrap=to_wrap, wrapped=self.wrapped, span=self.mock_span,
                                      instance=self.mock_instance, args=self.mock_args, kwargs=self.mock_kwargs,
                                      result=self.return_value)

        # Check if the correct log message is in the captured logs
        self.assertIn("type of span not found or incorrect written in entity json", log.output[0])


if __name__ == '__main__':
    unittest.main()
