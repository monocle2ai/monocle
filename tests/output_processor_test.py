
from monocle_apptrace.wrap_common import process_span
from monocle_apptrace.instrumentation.common.utils import load_output_processor
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

    def test_valid_output_processor(self):
        """Test case for valid output processor with type and attributes."""
        to_wrap ={
            "output_processor" :  {
                "type": "inference",
                "attributes": [
                    [
                        {
                            "attribute": "provider_name",
                            "accessor": "lambda args: 'example.com'"
                        },
                        {
                            "attribute": "inference_endpoint",
                            "accessor": "lambda args: 'https://example.com/'"
                        }
                    ]
                ]
            }
        }

        process_span(to_wrap, self.mock_span, self.mock_instance, self.mock_args, self.mock_kwargs, self.return_value)

        self.mock_span.set_attribute.assert_any_call("span.type", "inference")
        self.mock_span.set_attribute.assert_any_call("entity.count", 1)
        self.mock_span.set_attribute.assert_any_call("entity.1.provider_name", "example.com")
        self.mock_span.set_attribute.assert_any_call("entity.1.inference_endpoint", "https://example.com/")


    def test_output_processor_missing_span_type(self):
        """Test case when type is missing from output processor."""
        to_wrap ={
                "output_processor" : {
                "attributes": [
                    [
                        {
                            "attribute": "provider_name",
                            "accessor": "lambda args: 'example.com'"
                        },
                        {
                            "attribute": "inference_endpoint",
                            "accessor": "lambda args: 'https://example.com/'"
                        }
                    ]
                ]
            }
         }
        process_span(to_wrap, self.mock_span, self.mock_instance, self.mock_args, self.mock_kwargs, self.return_value)

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
        process_span(to_wrap, self.mock_span, self.mock_instance, self.mock_args, self.mock_kwargs, self.return_value)

        self.mock_span.set_attribute.assert_any_call("span.type", "inference")

    def test_empty_output_processor(self):
        """Test case for an empty output processor."""
        to_wrap={
            "output_processor":{}
        }
        process_span(to_wrap, self.mock_span, self.mock_instance, self.mock_args, self.mock_kwargs, self.return_value)

        # Log warning expected for incorrect format
        with self.assertLogs(level='WARNING') as log:
            process_span(to_wrap, self.mock_span, self.mock_instance, self.mock_args, self.mock_kwargs, self.return_value)

        # Check if the correct log message is in the captured logs
        self.assertIn("empty or entities json is not in correct format", log.output[0])

    def test_invalid_output_processor(self):
        """Test case for an invalid output processor format."""
        wrapper_method = {
                "output_processor":["src/monocle_apptrace/metamodel/maps/attributes/retrieval/langchain_entities1.json"]
        }
        attributes_config_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Log warning expected for incorrect format
        output_processor_file_path = wrapper_method["output_processor"][0]
        absolute_file_path = os.path.join(attributes_config_base_path, output_processor_file_path)

        with self.assertLogs(level='WARNING') as log:
            load_output_processor(wrapper_method, attributes_config_base_path)

        # Check if the correct log message is in the captured logs
        self.assertIn(f"Error: File not found at {absolute_file_path}", log.output[0])

if __name__ == '__main__':
    unittest.main()
