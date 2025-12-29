"""
Tests for Okahu client and CLI.
"""
import json
import os
import unittest
from unittest.mock import Mock, patch, MagicMock

from monocle_apptrace.exporters.okahu.okahu_client import OkahuClient


class TestOkahuClient(unittest.TestCase):
    """Test cases for OkahuClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        
    def test_client_initialization_with_api_key(self):
        """Test client initialization with explicit API key."""
        client = OkahuClient(api_key=self.api_key)
        self.assertEqual(client.api_key, self.api_key)
        self.assertIn("x-api-key", client.session.headers)
        self.assertEqual(client.session.headers["x-api-key"], self.api_key)
        
    def test_client_initialization_from_env(self):
        """Test client initialization from environment variable."""
        with patch.dict(os.environ, {"OKAHU_API_KEY": self.api_key}):
            client = OkahuClient()
            self.assertEqual(client.api_key, self.api_key)
    
    def test_client_initialization_without_api_key(self):
        """Test client initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                OkahuClient()
            self.assertIn("OKAHU_API_KEY not set", str(context.exception))
    
    @patch('requests.Session.get')
    def test_list_apps_success(self, mock_get):
        """Test successful listing of apps."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "apps": [
                {"id": "app1", "name": "App 1", "status": "active"},
                {"id": "app2", "name": "App 2", "status": "active"}
            ]
        }
        mock_get.return_value = mock_response
        
        client = OkahuClient(api_key=self.api_key)
        apps = client.list_apps()
        
        self.assertEqual(len(apps), 2)
        self.assertEqual(apps[0]["name"], "App 1")
        self.assertEqual(apps[1]["name"], "App 2")
        
        # Verify the correct endpoint was called
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn("/api/v1/apps", call_args[0][0])
    
    @patch('requests.Session.get')
    def test_list_apps_empty(self, mock_get):
        """Test listing apps when no apps exist."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"apps": []}
        mock_get.return_value = mock_response
        
        client = OkahuClient(api_key=self.api_key)
        apps = client.list_apps()
        
        self.assertEqual(len(apps), 0)
    
    @patch('requests.Session.get')
    def test_list_apps_api_error(self, mock_get):
        """Test handling of API errors when listing apps."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response
        
        client = OkahuClient(api_key=self.api_key)
        
        with self.assertRaises(Exception):
            client.list_apps()
    
    @patch('requests.Session.get')
    def test_get_app_success(self, mock_get):
        """Test successful retrieval of app details."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "app1",
            "name": "App 1",
            "status": "active",
            "workflow_name": "my-workflow"
        }
        mock_get.return_value = mock_response
        
        client = OkahuClient(api_key=self.api_key)
        app = client.get_app("app1")
        
        self.assertEqual(app["id"], "app1")
        self.assertEqual(app["name"], "App 1")
        
        # Verify the correct endpoint was called
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn("/api/v1/apps/app1", call_args[0][0])
    
    def test_context_manager(self):
        """Test client works as context manager."""
        with OkahuClient(api_key=self.api_key) as client:
            self.assertIsNotNone(client.session)
        
        # After context manager exits, session should be closed
        # We can't directly test this without mocking, but we verify no exception


if __name__ == '__main__':
    unittest.main()
