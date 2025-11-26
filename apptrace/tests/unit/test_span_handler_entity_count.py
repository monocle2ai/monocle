import unittest
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import Span

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler


class TestSpanHandlerEntityCount(unittest.TestCase):
    """Test cases for SpanHandler entity counting functionality."""
    
    def setUp(self):
        super().setUp()
        self.span_handler = SpanHandler()
        
        # Create a mock span with attributes dictionary
        self.mock_span = MagicMock(spec=Span)
        self.mock_span.attributes = {}
        self.mock_span.set_attribute = MagicMock(side_effect=self._mock_set_attribute)
        self.mock_span.parent = MagicMock()
        
        # Create a mock parent span
        self.mock_parent_span = MagicMock(spec=Span)
        self.mock_parent_span.attributes = {}
        
    def _mock_set_attribute(self, key, value):
        """Mock implementation of span.set_attribute that stores in attributes dict."""
        self.mock_span.attributes[key] = value
        
    def test_entity_count_with_two_entities_regular_execution(self):
        """Test entity.count = 2 when only provider and model entities are set during regular execution."""
        
        # Define output processor with 2 entities (provider and model)
        to_wrap = {
            'output_processor': {
                'type': 'inference.framework',
                'attributes': [
                    [
                        {
                            'attribute': 'type',
                            'accessor': lambda args: 'inference.openai'
                        },
                        {
                            'attribute': 'provider_name',
                            'accessor': lambda args: 'api.openai.com'
                        }
                    ],
                    [
                        {
                            'attribute': 'name',
                            'accessor': lambda args: 'gpt-3.5-turbo'
                        },
                        {
                            'attribute': 'type',
                            'accessor': lambda args: 'model.llm.gpt-3.5-turbo'
                        }
                    ]
                ]
            }
        }
        
        # Mock arguments
        args = []
        kwargs = {}
        result = MagicMock()
        instance = MagicMock()
        wrapped = MagicMock()
        
        # Mock get_scopes to return empty dict (no scope attributes)
        with patch('monocle_apptrace.instrumentation.common.span_handler.get_scopes', return_value={}):
            # Call hydrate_attributes during regular execution (not post_exec)
            self.span_handler.hydrate_attributes(
                to_wrap, wrapped, instance, args, kwargs, result, 
                self.mock_span, self.mock_parent_span, is_post_exec=False
            )
        
        # Verify entity.count is 2
        self.assertEqual(self.mock_span.attributes.get('entity.count'), 2)
        
        # Verify entity attributes are set correctly
        self.assertEqual(self.mock_span.attributes.get('entity.1.type'), 'inference.openai')
        self.assertEqual(self.mock_span.attributes.get('entity.1.provider_name'), 'api.openai.com')
        self.assertEqual(self.mock_span.attributes.get('entity.2.name'), 'gpt-3.5-turbo')
        self.assertEqual(self.mock_span.attributes.get('entity.2.type'), 'model.llm.gpt-3.5-turbo')
        
        # Verify no entity.3 attributes
        self.assertIsNone(self.mock_span.attributes.get('entity.3.name'))
        self.assertIsNone(self.mock_span.attributes.get('entity.3.type'))

    def test_entity_count_with_three_entities_including_tool_call(self):
        """Test entity.count = 3 when provider, model, and tool entities are set."""
        
        # First, set up the initial entities during regular execution
        to_wrap_regular = {
            'output_processor': {
                'type': 'inference.framework',
                'attributes': [
                    [
                        {
                            'attribute': 'type',
                            'accessor': lambda args: 'inference.openai'
                        },
                        {
                            'attribute': 'provider_name',
                            'accessor': lambda args: 'api.openai.com'
                        }
                    ],
                    [
                        {
                            'attribute': 'name',
                            'accessor': lambda args: 'gpt-4o-mini'
                        },
                        {
                            'attribute': 'type',
                            'accessor': lambda args: 'model.llm.gpt-4o-mini'
                        }
                    ]
                ]
            }
        }
        
        # Call hydrate_attributes during regular execution
        self.span_handler.hydrate_attributes(
            to_wrap_regular, MagicMock(), MagicMock(), [], {}, MagicMock(),
            self.mock_span, self.mock_parent_span, is_post_exec=False
        )
        
        # Now simulate post_execution with tool attributes
        to_wrap_post_exec = {
            'output_processor': {
                'type': 'inference.framework',
                'attributes': [
                    [
                        {
                            'attribute': 'name',
                            'accessor': lambda args: 'get_weather',
                            'phase': 'post_execution'
                        },
                        {
                            'attribute': 'type',
                            'accessor': lambda args: 'tool.function',
                            'phase': 'post_execution'
                        }
                    ]
                ]
            }
        }
        
        # Call hydrate_attributes during post_execution
        self.span_handler.hydrate_attributes(
            to_wrap_post_exec, MagicMock(), MagicMock(), [], {}, MagicMock(),
            self.mock_span, self.mock_parent_span, is_post_exec=True
        )
        
        # Verify entity.count is 3
        self.assertEqual(self.mock_span.attributes.get('entity.count'), 3)
        
        # Verify all entity attributes are set correctly
        self.assertEqual(self.mock_span.attributes.get('entity.1.type'), 'inference.openai')
        self.assertEqual(self.mock_span.attributes.get('entity.1.provider_name'), 'api.openai.com')
        self.assertEqual(self.mock_span.attributes.get('entity.2.name'), 'gpt-4o-mini')
        self.assertEqual(self.mock_span.attributes.get('entity.2.type'), 'model.llm.gpt-4o-mini')
        self.assertEqual(self.mock_span.attributes.get('entity.3.name'), 'get_weather')
        self.assertEqual(self.mock_span.attributes.get('entity.3.type'), 'tool.function')

    def test_entity_count_with_empty_entity_processor(self):
        """Test entity.count when entity processor exists but has no valid attributes."""

        # Define output processor with empty/invalid attributes
        to_wrap = {
            'output_processor': {
                'type': 'inference.framework',
                'attributes': [
                    [
                        {
                            'attribute': 'type',
                            'accessor': lambda args: None  # Returns None
                        }
                    ],
                    [
                        {
                            'attribute': 'name',
                            'accessor': lambda args: ''  # Returns empty string
                        }
                    ]
                ]
            }
        }

        # Mock arguments using the same pattern as other tests
        args = []
        kwargs = {}
        result = MagicMock()
        instance = MagicMock()
        wrapped = MagicMock()

        # Mock get_scopes to return empty dict (no scope attributes)
        with patch('monocle_apptrace.instrumentation.common.span_handler.get_scopes', return_value={}):
            # Call hydrate_attributes
            self.span_handler.hydrate_attributes(
                to_wrap, wrapped, instance, args, kwargs, result,
                self.mock_span, self.mock_parent_span, is_post_exec=False
            )

        # Verify entity.count is not set when no valid entities are processed
        # (The implementation only sets entity.count when span_index > 0)
        self.assertNotIn('entity.count', self.mock_span.attributes)

    def test_entity_count_with_mixed_valid_invalid_entities(self):
        """Test entity.count when some entities are valid and others are not."""
        
        # Define output processor with mixed valid/invalid entities
        to_wrap = {
            'output_processor': {
                'type': 'inference.framework',
                'attributes': [
                    [
                        # Valid entity
                        {
                            'attribute': 'type',
                            'accessor': lambda args: 'inference.openai'
                        }
                    ],
                    [
                        # Invalid entity (returns None)
                        {
                            'attribute': 'name',
                            'accessor': lambda args: None
                        }
                    ],
                    [
                        # Valid entity
                        {
                            'attribute': 'name',
                            'accessor': lambda args: 'gpt-4'
                        }
                    ]
                ]
            }
        }
        
        # Call hydrate_attributes
        self.span_handler.hydrate_attributes(
            to_wrap, MagicMock(), MagicMock(), [], {}, MagicMock(),
            self.mock_span, self.mock_parent_span, is_post_exec=False
        )
        
        # Verify entity.count is 2 (only valid entities counted)
        self.assertEqual(self.mock_span.attributes.get('entity.count'), 2)
        self.assertEqual(self.mock_span.attributes.get('entity.1.type'), 'inference.openai')
        self.assertEqual(self.mock_span.attributes.get('entity.2.name'), 'gpt-4')

    def test_entity_count_post_execution_index_calculation(self):
        """Test that post_execution correctly finds the next available entity index."""
        
        # Pre-populate some entities
        self.mock_span.attributes.update({
            'entity.1.type': 'inference.openai',
            'entity.1.provider_name': 'api.openai.com',
            'entity.2.name': 'gpt-4',
            'entity.2.type': 'model.llm.gpt-4'
        })
        
        # Define post_execution processor for tool entity
        to_wrap_post_exec = {
            'output_processor': {
                'type': 'inference.framework',
                'attributes': [
                    [
                        {
                            'attribute': 'name',
                            'accessor': lambda args: 'function_call',
                            'phase': 'post_execution'
                        },
                        {
                            'attribute': 'type',
                            'accessor': lambda args: 'tool.function',
                            'phase': 'post_execution'
                        }
                    ]
                ]
            }
        }
        
        # Call hydrate_attributes during post_execution
        self.span_handler.hydrate_attributes(
            to_wrap_post_exec, MagicMock(), MagicMock(), [], {}, MagicMock(),
            self.mock_span, self.mock_parent_span, is_post_exec=True
        )
        
        # Verify entity.count is 3 and tool entity is at entity.3
        self.assertEqual(self.mock_span.attributes.get('entity.count'), 3)
        self.assertEqual(self.mock_span.attributes.get('entity.3.name'), 'function_call')
        self.assertEqual(self.mock_span.attributes.get('entity.3.type'), 'tool.function')

    def test_entity_count_no_output_processor(self):
        """Test entity.count when no output_processor is defined."""
        
        to_wrap = {}  # No output_processor
        
        # Call hydrate_attributes
        self.span_handler.hydrate_attributes(
            to_wrap, MagicMock(), MagicMock(), [], {}, MagicMock(),
            self.mock_span, self.mock_parent_span, is_post_exec=False
        )
        
        # Verify no entity.count is set
        self.assertNotIn('entity.count', self.mock_span.attributes)

    def test_entity_count_with_exception_in_accessor(self):
        """Test entity.count behavior when accessor throws an exception."""
        
        def failing_accessor(args):
            raise Exception("Test exception")
        
        # Define output processor with failing accessor
        to_wrap = {
            'output_processor': {
                'type': 'inference.framework',
                'attributes': [
                    [
                        {
                            'attribute': 'type',
                            'accessor': failing_accessor
                        }
                    ],
                    [
                        {
                            'attribute': 'name',
                            'accessor': lambda args: 'valid_model'
                        }
                    ]
                ]
            }
        }
        
        # Call hydrate_attributes
        result = self.span_handler.hydrate_attributes(
            to_wrap, MagicMock(), MagicMock(), [], {}, MagicMock(),
            self.mock_span, self.mock_parent_span, is_post_exec=False
        )
        
        # Verify entity.count is 1 (only valid entity counted)
        self.assertEqual(self.mock_span.attributes.get('entity.count'), 1)
        self.assertEqual(self.mock_span.attributes.get('entity.1.name'), 'valid_model')


if __name__ == '__main__':
    unittest.main()