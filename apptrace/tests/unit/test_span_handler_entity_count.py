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


    def test_entity_count_with_mixed_phase_attributes_in_entity(self):
            """Test entity.count when one attribute in an entity has 'phase': 'post_execution' and others don't.

            When an entity group has mixed phase attributes:
            - Attributes WITHOUT 'phase': 'post_execution' are set during REGULAR execution
            - Attributes WITH 'phase': 'post_execution' are set during POST_EXECUTION
            - Both are added to the SAME entity index

            Expected:
            - Regular execution: entity.1.provider_name set
            - Post execution: entity.1.type added (same entity)
            - Final: entity.count = 2
            """

            # Define output processor with mixed phase attributes in same entity
            to_wrap = {
                'output_processor': {
                    'type': 'inference.framework',
                    'attributes': [
                        [
                            {
                                'attribute': 'type',
                                'accessor': lambda args: 'inference.openai',
                                'phase': 'post_execution'  # Will be set during post_execution
                            },
                            {
                                'attribute': 'provider_name',
                                'accessor': lambda args: 'api.openai.com'  # Will be set during regular execution
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

            # Call hydrate_attributes during regular execution
            self.span_handler.hydrate_attributes(
                to_wrap, MagicMock(), MagicMock(), [], {}, MagicMock(),
                self.mock_span, self.mock_parent_span, is_post_exec=False
            )

            # After regular execution: entity.1 has provider_name only (type has post_execution phase)
            self.assertEqual(self.mock_span.attributes.get('entity.count'), 2)
            self.assertEqual(self.mock_span.attributes.get('entity.1.provider_name'), 'api.openai.com')
            self.assertIsNone(self.mock_span.attributes.get('entity.1.type'),
                             "entity.1.type should not be set yet (has post_execution phase)")
            self.assertEqual(self.mock_span.attributes.get('entity.2.name'), 'gpt-3.5-turbo')
            self.assertEqual(self.mock_span.attributes.get('entity.2.type'), 'model.llm.gpt-3.5-turbo')

            # Call hydrate_attributes during post_execution
            self.span_handler.hydrate_attributes(
                to_wrap, MagicMock(), MagicMock(), [], {}, MagicMock(),
                self.mock_span, self.mock_parent_span, is_post_exec=True
            )

            # After post_execution: entity.1.type is now added to the same entity
            self.assertEqual(self.mock_span.attributes.get('entity.count'), 2)
            self.assertEqual(self.mock_span.attributes.get('entity.1.type'), 'inference.openai',
                            "entity.1.type should now be set")
            self.assertEqual(self.mock_span.attributes.get('entity.1.provider_name'), 'api.openai.com',
                            "entity.1.provider_name should still be present")


if __name__ == '__main__':
    unittest.main()