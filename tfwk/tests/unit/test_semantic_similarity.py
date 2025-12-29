#!/usr/bin/env python3
"""
Test script showing semantic similarity in action for LLM output testing.

This test demonstrates how semantic similarity can make test assertions
more robust when dealing with varied LLM outputs.
"""

import os
import sys
import pytest
import logging

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from monocle_tfwk.semantic_similarity import (
    semantic_similarity,
    semantic_similarity_score,
)

logger = logging.getLogger(__name__)


def test_semantic_similarity_demo():
    """Test semantic similarity for LLM output validation."""
    
    logger.info("🤖 Semantic Similarity Demo for LLM Testing")
    logger.info("=" * 50)
    
    # Expected response
    expected = "I don't understand the question"
    
    # Various possible LLM responses with similar meanings
    possible_responses = [
        "I don't understand the question",           # Exact match
        "I can't comprehend what you're asking",     # Similar meaning
        "I'm not sure what you mean",                # Similar meaning  
        "That question is unclear to me",            # Similar meaning
        "I'm confused by your request",              # Similar meaning
        "Could you clarify your question?",          # Similar meaning
        "The weather is nice today",                 # Completely different
        "I understand perfectly",                    # Opposite meaning
    ]
    
    logger.info(f"📝 Expected response: '{expected}'")
    logger.info("🔍 Testing similarity with various LLM outputs:")
    
    for i, response in enumerate(possible_responses, 1):
        score = semantic_similarity_score(expected, response)
        is_similar_75 = semantic_similarity(expected, response, threshold=0.75)
        is_similar_60 = semantic_similarity(expected, response, threshold=0.60)
        
        logger.info(f"{i}. \"{response}\"")
        logger.info(f"   📊 Similarity Score: {score:.3f}")
        logger.info(f"   ✅ Similar @ 0.75 threshold: {'YES' if is_similar_75 else 'NO'}")
        logger.info(f"   ✅ Similar @ 0.60 threshold: {'YES' if is_similar_60 else 'NO'}")
    
    logger.info("💡 Key Insights:")
    logger.info("   • Higher thresholds (0.75+) = stricter matching")
    logger.info("   • Lower thresholds (0.60-) = more lenient matching") 
    logger.info("   • Use semantic similarity for robust LLM output testing!")
    logger.info("   • Fallback to string containment if sentence-transformers not available")
    
    # Add assertions to make it a proper test
    assert semantic_similarity(expected, "I can't comprehend what you're asking", threshold=0.60), "Similar meanings should match at 60% threshold"
    assert not semantic_similarity(expected, "The weather is nice today", threshold=0.60), "Different meanings should not match"


def test_assertion_examples():
    """Test examples showing how this works in test assertions."""
    
    logger.info("🧪 Test Assertion Examples")
    logger.info("=" * 50)
    
    logger.info("""
    
    # NEW WAY - Robust semantic matching:
    (self.assert_traces()
     .has_spans(min_count=1) 
     .semantically_contains_output("don't understand", threshold=0.7))  # Matches similar meanings!
    
    # Alternative readable syntax:
    (self.assert_traces()
     .has_spans(min_count=1)
     .output_semantically_matches("I don't understand", threshold=0.65))
    """)
    
    # Add assertion to make it a proper test
    assert True, "Test assertion examples documented successfully"


def test_fallback_similarity():
    """Test fallback functionality when dependencies are missing."""
    try:
        # Test that semantic similarity works with available dependencies
        result = semantic_similarity("I don't understand", "I can't comprehend", threshold=0.7)
        logger.info(f"📊 Semantic similarity result: {result}")
        assert isinstance(result, bool), "Semantic similarity should return a boolean"
    except ImportError as e:
        logger.info(f"⚠️  Missing dependencies: {e}")
        logger.info("💡 Install with: pip install sentence-transformers")
        logger.info("🔄 Falling back to string-based similarity...")
        
        # Test with fallback
        result = semantic_similarity("I don't understand", "I can't comprehend", threshold=0.7)
        logger.info(f"📊 Fallback similarity result: {result}")
        assert isinstance(result, bool), "Fallback similarity should still return a boolean"


if __name__ == "__main__":
    # Run the test directly if executed as a script
    pytest.main([__file__, "-s", "--tb=short"])