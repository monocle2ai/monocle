"""
Okahu API Client for querying and managing applications on Okahu platform.
"""
import json
import logging
import os
from typing import Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

OKAHU_API_BASE_URL = "https://api.okahu.co"


class OkahuClient:
    """Client for interacting with Okahu API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize Okahu API client.
        
        Args:
            api_key: Okahu API key. If not provided, will use OKAHU_API_KEY environment variable.
            base_url: Base URL for Okahu API. Defaults to production API endpoint.
            timeout: Request timeout in seconds.
        
        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
        self.api_key = api_key or os.environ.get("OKAHU_API_KEY")
        if not self.api_key:
            raise ValueError("OKAHU_API_KEY not set. Please provide an API key or set the OKAHU_API_KEY environment variable.")
        
        self.base_url = base_url or os.environ.get("OKAHU_API_BASE_URL", OKAHU_API_BASE_URL)
        self.timeout = timeout
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        })
    
    def list_apps(self) -> List[Dict]:
        """
        List all applications registered on Okahu platform.
        
        Returns:
            List of application dictionaries containing app metadata.
            
        Raises:
            requests.RequestException: If the API request fails.
        """
        try:
            url = f"{self.base_url}/api/v1/apps"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            apps = data.get("apps", [])
            
            logger.info(f"Successfully retrieved {len(apps)} apps from Okahu")
            return apps
            
        except requests.RequestException as e:
            logger.error(f"Failed to list apps from Okahu: {e}")
            raise
    
    def get_app(self, app_id: str) -> Dict:
        """
        Get details for a specific application.
        
        Args:
            app_id: The application ID or workflow name.
            
        Returns:
            Application details dictionary.
            
        Raises:
            requests.RequestException: If the API request fails.
        """
        try:
            url = f"{self.base_url}/api/v1/apps/{app_id}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Failed to get app {app_id} from Okahu: {e}")
            raise
    
    def close(self):
        """Close the session."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
