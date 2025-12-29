"""
Integration test for Okahu CLI demonstrating end-to-end functionality.
"""
import json
import os
import sys
import unittest
from io import StringIO
from unittest.mock import patch, Mock

from monocle_apptrace.exporters.okahu import OkahuClient


class TestOkahuCLIIntegration(unittest.TestCase):
    """Integration tests for Okahu CLI and client."""
    
    def setUp(self):
        """Set up test environment."""
        self.api_key = "test_api_key_12345"
    
    @patch('requests.Session.get')
    def test_client_list_apps_integration(self, mock_get):
        """Test client list apps integration."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "apps": [
                {
                    "id": "app-123",
                    "name": "My AI Agent",
                    "status": "active",
                    "workflow_name": "ai-agent-workflow"
                },
                {
                    "id": "app-456",
                    "name": "Chat Bot",
                    "status": "inactive",
                    "workflow_name": "chatbot-workflow"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Use client
        with OkahuClient(api_key=self.api_key) as client:
            apps = client.list_apps()
        
        # Verify results
        self.assertEqual(len(apps), 2)
        self.assertEqual(apps[0]["name"], "My AI Agent")
        self.assertEqual(apps[0]["workflow_name"], "ai-agent-workflow")
        self.assertEqual(apps[1]["name"], "Chat Bot")
        self.assertEqual(apps[1]["status"], "inactive")
    
    @patch('requests.Session.get')
    def test_client_get_app_integration(self, mock_get):
        """Test client get app integration."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "app-999",
            "name": "Detailed App",
            "status": "active",
            "workflow_name": "my-workflow",
            "created_at": "2025-01-01T00:00:00Z",
            "traces_count": 150,
            "last_trace_at": "2025-01-15T12:00:00Z"
        }
        mock_get.return_value = mock_response
        
        # Use client
        with OkahuClient(api_key=self.api_key) as client:
            app = client.get_app("app-999")
        
        # Verify results
        self.assertEqual(app["id"], "app-999")
        self.assertEqual(app["name"], "Detailed App")
        self.assertEqual(app["workflow_name"], "my-workflow")
        self.assertEqual(app["traces_count"], 150)
    
    @patch('requests.Session.get')
    def test_empty_apps_list(self, mock_get):
        """Test handling of empty apps list."""
        # Mock API response with no apps
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"apps": []}
        mock_get.return_value = mock_response
        
        # Use client
        with OkahuClient(api_key=self.api_key) as client:
            apps = client.list_apps()
        
        # Verify results
        self.assertEqual(len(apps), 0)
        self.assertIsInstance(apps, list)


if __name__ == '__main__':
    unittest.main()
