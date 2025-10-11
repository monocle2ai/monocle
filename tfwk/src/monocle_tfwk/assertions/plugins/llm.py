"""
LLM-specific assertion plugins for validating LLM calls and inference.

This module contains plugins that provide assertions specifically for validating
LLM inference calls, token usage, costs, and model interactions.
"""
from typing import TYPE_CHECKING, Any, List

from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin

if TYPE_CHECKING:
    from monocle_tfwk.assertions.trace_assertions import TraceAssertions


@plugin
class LLMAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing LLM-specific assertion methods."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "llm"
    
    def llm_calls(self) -> List[Any]:
        """Get list of LLM call spans."""
        def is_llm_span(span):
            """Detect if a span represents an LLM call using multiple strategies."""
            # Strategy 1: Check span.type for inference
            span_type = span.attributes.get("span.type")
            if span_type == "inference":
                return True
            
            # Strategy 2: Check for entity.*.type == inference.*
            for key, value in span.attributes.items():
                if (key.startswith("entity.") and 
                    key.endswith(".type") and 
                    isinstance(value, str) and 
                    value.startswith("inference.")):
                    return True
            
            # Strategy 3: Check for LLM-specific attributes
            llm_indicators = [
                "llm.model", "llm.provider", "llm.request", "llm.response",
                "openai.model", "anthropic.model", "cohere.model",
                "model.name", "model.provider"
            ]
            
            for indicator in llm_indicators:
                if indicator in span.attributes:
                    return True
            
            # Strategy 4: Check entity types for LLM models
            for key, value in span.attributes.items():
                if (key.startswith("entity.") and 
                    key.endswith(".type") and 
                    isinstance(value, str) and 
                    ("model.llm" in value or "llm" in value.lower())):
                    return True
            
            # Strategy 5: Check span name for LLM indicators
            if span.name and any(indicator in span.name.lower() for indicator in 
                               ["llm", "openai", "anthropic", "cohere", "inference", "completion"]):
                return True
            
            return False
        
        return [span for span in self._current_spans if is_llm_span(span)]
    

    
    def assert_llm_calls(self, count: int = None, min_count: int = None) -> 'TraceAssertions':
        """Assert LLM call count conditions."""
        # Count LLM spans using our enhanced detection logic
        def is_llm_span(span):
            """Detect if a span represents an LLM call using multiple strategies."""
            # Strategy 1: Check span.type for inference
            span_type = span.attributes.get("span.type")
            if span_type == "inference":
                return True
            
            # Strategy 2: Check for entity.*.type == inference.*
            for key, value in span.attributes.items():
                if (key.startswith("entity.") and 
                    key.endswith(".type") and 
                    isinstance(value, str) and 
                    value.startswith("inference.")):
                    return True
            
            # Strategy 3: Check for LLM-specific attributes
            llm_indicators = [
                "llm.model", "llm.provider", "llm.request", "llm.response",
                "openai.model", "anthropic.model", "cohere.model",
                "model.name", "model.provider"
            ]
            
            for indicator in llm_indicators:
                if indicator in span.attributes:
                    return True
            
            # Strategy 4: Check entity types for LLM models
            for key, value in span.attributes.items():
                if (key.startswith("entity.") and 
                    key.endswith(".type") and 
                    isinstance(value, str) and 
                    ("model.llm" in value or "llm" in value.lower())):
                    return True
            
            # Strategy 5: Check span name for LLM indicators
            if span.name and any(indicator in span.name.lower() for indicator in 
                               ["llm", "openai", "anthropic", "cohere", "inference", "completion"]):
                return True
            
            return False
        
        llm_count = sum(1 for span in self._current_spans if is_llm_span(span))
        
        if count is not None:
            assert llm_count == count, f"Expected exactly {count} LLM calls, found {llm_count}"
        elif min_count is not None:
            assert llm_count >= min_count, f"Expected at least {min_count} LLM calls, found {llm_count}"
            
        return self
    
    def assert_min_llm_calls(self, min_calls: int) -> 'TraceAssertions':
        """Assert minimum number of LLM calls."""
        # Count LLM spans using our detection logic
        def is_llm_span(span):
            """Detect if a span represents an LLM call using multiple strategies."""
            # Strategy 1: Check span.type for inference
            span_type = span.attributes.get("span.type")
            if span_type == "inference":
                return True
            
            # Strategy 2: Check for entity.*.type == inference.*
            for key, value in span.attributes.items():
                if (key.startswith("entity.") and 
                    key.endswith(".type") and 
                    isinstance(value, str) and 
                    value.startswith("inference.")):
                    return True
            
            # Strategy 3: Check for LLM-specific attributes
            llm_indicators = [
                "llm.model", "llm.provider", "llm.request", "llm.response",
                "openai.model", "anthropic.model", "cohere.model",
                "model.name", "model.provider"
            ]
            
            for indicator in llm_indicators:
                if indicator in span.attributes:
                    return True
            
            # Strategy 4: Check entity types for LLM models
            for key, value in span.attributes.items():
                if (key.startswith("entity.") and 
                    key.endswith(".type") and 
                    isinstance(value, str) and 
                    ("model.llm" in value or "llm" in value.lower())):
                    return True
            
            # Strategy 5: Check span name for LLM indicators
            if span.name and any(indicator in span.name.lower() for indicator in 
                               ["llm", "openai", "anthropic", "cohere", "inference", "completion"]):
                return True
            
            return False
        
        llm_count = sum(1 for span in self._current_spans if is_llm_span(span))
        assert llm_count >= min_calls, f"Expected at least {min_calls} LLM calls, found {llm_count}"
        return self
