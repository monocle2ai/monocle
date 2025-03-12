import os
import requests
import json
from typing import List, Dict, Any, Optional


class OpenAIClient:
    """Client for interacting with OpenAI's Chat API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env variable.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Either pass it explicitly or set OPENAI_API_KEY environment variable.")
        
        self.base_url = "https://api.openai.com/v1"
    
    def chat(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", 
             temperature: float = 0.7, max_tokens: Optional[int] = None,
             top_p: float = 1.0, frequency_penalty: float = 0.0,
             presence_penalty: float = 0.0) -> Dict[str, Any]:
        """
        Call OpenAI's chat completion API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: OpenAI model identifier to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalty for token frequency
            presence_penalty: Penalty for token presence
        
        Returns:
            Complete API response including content and metadata
        """
        url = f"{self.base_url}/chat/completions"
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Make API request
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API request failed with status {response.status_code}: {response.text}")
        
        return response.json()

    def format_messages(self, system_prompts: List[str], user_prompts: List[str]) -> List[Dict[str, str]]:
        """
        Format system and user prompts into the message format required by OpenAI API.
        
        Args:
            system_prompts: List of system prompts
            user_prompts: List of user prompts
        
        Returns:
            List of formatted message dictionaries
        """
        messages = []
        
        # Add system messages
        for system_prompt in system_prompts:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user messages
        for user_prompt in user_prompts:
            messages.append({"role": "user", "content": user_prompt})
            
        return messages
