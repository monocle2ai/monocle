"""
LLM client implementations for various providers.

This module contains client implementations that handle communication
with different LLM providers through standardized interfaces.
"""

import logging
from typing import Any, Dict, Optional

import litellm
from litellm import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    RateLimitError,
)

# from models.models import FinishDetails, FinishType, ReasonCode

logger = logging.getLogger(__name__)


class LiteLLMClient:
    """
    Professional LLM client using LiteLLM for multi-provider support.

    This client provides a unified interface for interacting with various LLM providers
    while handling errors, content filtering, and response validation consistently.
    """



    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            api_key: Optional API key. If not provided, will use environment variables.
        """
        self.api_key = api_key

    def _create_response_dict(
            self,
            content: Optional[str],
    ) -> Dict[str, Any]:
        """Create a standardized response dictionary."""
        return {
            'content': content,
            'finish_type': 'success',
            'finish_details': None
        }

    def get_completion(
            self,
            model: Optional[str] = None,
            eval_prompt: Optional[str] = None,
            prompt: Optional[str] = None,
            response_format: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Get completion from LLM with comprehensive error handling.

        Args:
            model: Model identifier (e.g., 'gpt-4', 'claude-3-sonnet')
            eval_prompt: System prompt template
            prompt: User prompt
            response_format: Structured output format specification
            **kwargs: Additional parameters for the LLM API

        Returns:
            Dictionary containing content, finish_type, and finish_details

        Raises:
            TypeError: If required parameters are missing
            Various LiteLLM exceptions: For authentication, connection, or other API errors
        """
        # Validation
        #gpt-4o-mini | azure/gpt-4o-mini | bedrock/anthropic.claude-3-sonnet-20240229-v1:0

        if model is None and not hasattr(self, 'default_model'):
            raise TypeError("Model must be specified either as an argument or as a default.")
        if eval_prompt is None:
            raise TypeError("Eval prompt must be provided.")
        if prompt is None:
            raise TypeError("User prompt must be provided.")
        if response_format is None:
            raise TypeError("Response format must be specified.")

        final_model = model if model is not None else 'gpt-4o-mini'

        # Prepare messages
        messages = []
        if eval_prompt:
            messages.append({"role": "system", "content": eval_prompt})
        if prompt:
            messages.append({"role": "user", "content": prompt})

        completion_params = {
            "model": final_model,
            "messages": messages,
            **kwargs
        }

        if response_format:
            completion_params["response_format"] = response_format

        try:
            if self.api_key is not None:
                # Fallback option - litellm will read from environment variables
                litellm.api_key = self.api_key

            logger.info(f"Sending LLM request with parameters: {completion_params}")
            completion = litellm.completion(**completion_params)

            return self._process_completion_response(completion)

        except (AuthenticationError, APIConnectionError, Exception) as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise e

    def _process_completion_response(self, completion) -> Dict[str, Any]:
        """
        Process the completion response and determine finish type and details.

        Args:
            completion: The completion response from litellm

        Returns:
            Response dictionary with content, finish_type, and finish_details
        """


        content = completion.choices[0].message.content
        finish_reason = completion.choices[0].finish_reason




        return self._create_response_dict(content)

