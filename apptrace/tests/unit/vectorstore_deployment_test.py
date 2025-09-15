import logging
import unittest
from unittest.mock import Mock

from monocle_apptrace.instrumentation.metamodel.haystack._helper import (
    get_vectorstore_deployment,
)

logger = logging.getLogger(__name__)

class TestGetVectorstoreDeployment(unittest.TestCase):
    def setUp(self):
        """Set up mock objects and variables for testing."""
        # Mock data for testing
        self.mock_map_with_client_settings = {
            '_client_settings': Mock(),
        }
        self.mock_map_with_client_settings['_client_settings'].host = 'localhost'
        self.mock_map_with_client_settings['_client_settings'].port = '50052'

        self.mock_map_with_seed_connections = {
            'client': {
                'transport': {
                    'seed_connections': [
                        Mock()
                    ]
                }
            }
        }
        self.mock_map_with_seed_connections['client']['transport']['seed_connections'][0].host = 'https://search-opensearch.amazonaws.com'

        self.mock_object_with_endpoint = Mock()
        self.mock_object_with_endpoint.client = Mock()
        self.mock_object_with_endpoint.client._endpoint = 'https://search-opensearch.amazonaws.com'

        self.mock_object_with_host_and_port = Mock()
        self.mock_object_with_host_and_port.host = 'localhost'
        self.mock_object_with_host_and_port.port = '50052'

    def test_host_and_port(self):
        """Test case when 'host' or 'port' is missing, or both are missing."""
        # Test case where both host and port are available
        self.mock_object_with_host_and_port.host = 'localhost'
        self.mock_object_with_host_and_port.port = '50052'
        result = get_vectorstore_deployment(self.mock_object_with_host_and_port)
        self.assertEqual(result, 'localhost:50052')

        # Test case where only host is available (port is missing)
        self.mock_object_with_host_and_port.host = 'localhost'
        self.mock_object_with_host_and_port.port = None
        result = get_vectorstore_deployment(self.mock_object_with_host_and_port)
        self.assertEqual(result, 'localhost')

        # Test case where neither host nor port is available
        self.mock_object_with_host_and_port.host = None
        self.mock_object_with_host_and_port.port = None
        result = get_vectorstore_deployment(self.mock_object_with_host_and_port)
        self.assertIsNone(result)

    def test_with_seed_connections(self):
        """Test case when 'seed_connections' key is present."""
        result = get_vectorstore_deployment(self.mock_map_with_seed_connections)
        self.assertEqual(result, 'https://search-opensearch.amazonaws.com')

    def test_with_object_endpoint(self):
        """Test case when object has '_endpoint' in the client."""
        result = get_vectorstore_deployment(self.mock_object_with_endpoint)
        self.assertEqual(result, 'https://search-opensearch.amazonaws.com')


if __name__ == '__main__':
    unittest.main()
