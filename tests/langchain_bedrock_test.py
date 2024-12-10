import unittest
from unittest.mock import patch, MagicMock
from monocle_apptrace.wrap_common import process_span
import requests


class TestProcessSpan(unittest.TestCase):

    @patch.object(requests.Session, 'post')
    def test_process_span_with_mocked_eval(self, mock_eval):
        mock_eval.side_effect = lambda expression: {
            "lambda arguments: arguments['kwargs']['provider_name'] or arguments['instance'].provider": "value1",
            "lambda arguments: extract_messages(arguments['args']": "What is Task Decomposition?"
        }.get(expression, None)

        span = MagicMock()
        span.set_attribute = MagicMock()
        instance = MagicMock()

        args = (MagicMock(messages=[
            MagicMock(content="System message", type="system"),
            MagicMock(content="What is Task Decomposition?", type="user")
        ]), {})
        kwargs = {"key1": "value1", "provider_name": "value1"}
        return_value = "test_return_value"

        # Define wrap_attributes with attributes and events to process
        wrap_attributes = {
            "output_processor": {
                "type": "inference",
                "attributes": [
                    [
                        {
                            "attribute": "provider_name",
                            "accessor": "lambda arguments: arguments['kwargs']['provider_name'] or arguments['instance'].provider"
                        }
                    ]
                ],
                "events": [
                    {
                        "name": "data.input",
                        "attributes": [
                            {
                                "attribute": "user",
                                "accessor": "lambda arguments: extract_messages(arguments['args'])"
                            }
                        ]
                    }
                ]
            }
        }

        process_span(to_wrap=wrap_attributes, span=span, instance=instance, args=args, kwargs=kwargs,
                     return_value=return_value)

        # Verify the span and events attributes
        span.set_attribute.assert_any_call("entity.count", 1)
        span.set_attribute.assert_any_call("span.type", "inference")
        span.set_attribute.assert_any_call("entity.1.provider_name", "value1")
        span.add_event.assert_any_call(name="data.input", attributes={'user': ["{'system': 'System message'}", "{'user': 'What is Task Decomposition?'}"]})

if __name__ == '__main__':
    unittest.main()
