"""
Prompt loading utilities for LLM evaluation.

This module handles loading and converting JSON templates to Pydantic models
for structured LLM outputs and evaluation workflows.
"""

import logging
from typing import Any, Dict, Type

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


class PromptLoader:
    """Utility class for loading and converting evaluation templates to structured prompts."""

    @staticmethod
    def _create_eval_response_model(template: Dict[str, Any]) -> tuple[str, Type[BaseModel]]:
        try:
            # Extract field definitions
            fields = template.get("structure_output", {})

            # Create field definitions for Pydantic
            # https://docs.pydantic.dev/2.5/errors/usage_errors/#model-field-missing-annotation
            field_definitions = {
                field_name: (str, Field(...))  # Field(...) indicates required field
                for field_name in fields.keys()
            }
            logger.info(f"Creating Pydantic model with fields: {field_definitions}")

            # Create and return the model
            response_model_name = template.get("name", "EvalResult")
            return response_model_name, create_model(response_model_name, **field_definitions)

        except Exception as e:
            logger.error(f"Failed to create evaluation response model: {str(e)}")
            raise

    @classmethod
    def get_chat_prompt_template(cls, template_config: Dict[str, Any]) -> tuple[str, Type[BaseModel], str]:
        template_name, response_model = cls._create_eval_response_model(template_config)
        eval_prompt = template_config.get("eval_prompt", "")
        return template_name, response_model, eval_prompt
