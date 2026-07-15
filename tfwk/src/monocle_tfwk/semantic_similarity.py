"""
Semantic similarity utilities for fuzzy matching in test assertions.

This module provides semantic similarity checking for LLM outputs using 
sentence transformers, allowing for more robust test assertions that don't
rely on exact string matching.
"""

import warnings
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    np = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configuration
DEFAULT_MODEL_NAME = "sentence-transformers/all-roberta-large-v1"  # Highest quality model
DEFAULT_SIMILARITY_THRESHOLD = 0.75

class SemanticSimilarityChecker:
    """Semantic similarity checker using sentence transformers."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            warnings.warn(
                "sentence-transformers not available. Semantic similarity features will be disabled. "
                "Install with: pip install sentence-transformers"
            )
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Only load if needed and suppress progress bars in tests
            self._model = SentenceTransformer(self.model_name, device='cpu')
        return self._model
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score between 0 and 1 (1 being identical meaning)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            # Fallback to simple string containment check
            return 1.0 if text2.lower() in text1.lower() or text1.lower() in text2.lower() else 0.0
        
        if not text1 or not text2:
            return 0.0
            
        # Generate embeddings
        embeddings = self.model.encode([text1, text2], convert_to_numpy=True, normalize_embeddings=True)
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    def are_similar(self, text1: str, text2: str, threshold: Optional[float] = None) -> bool:
        """
        Check if two texts are semantically similar.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            threshold: Similarity threshold (uses instance default if None)
            
        Returns:
            True if texts are semantically similar above threshold
        """
        if threshold is None:
            threshold = self.threshold
            
        similarity_score = self.calculate_similarity(text1, text2)
        return similarity_score >= threshold
    
    def find_similar_in_list(self, target_text: str, text_list: List[str], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Find texts in a list that are similar to the target text.
        
        Args:
            target_text: Text to find similarities for
            text_list: List of texts to search in
            threshold: Similarity threshold (uses instance default if None)
            
        Returns:
            List of dictionaries with 'text', 'score', and 'similar' keys
        """
        if threshold is None:
            threshold = self.threshold
            
        results = []
        for text in text_list:
            score = self.calculate_similarity(target_text, text)
            results.append({
                'text': text,
                'score': score,
                'similar': score >= threshold
            })
        
        return results


# Global instance for convenience
_default_checker = None

def get_default_checker() -> SemanticSimilarityChecker:
    """Get or create the default semantic similarity checker."""
    global _default_checker
    if _default_checker is None:
        _default_checker = SemanticSimilarityChecker()
    return _default_checker

def semantic_similarity(text1: str, text2: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> bool:
    """
    Convenience function to check semantic similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text  
        threshold: Similarity threshold
        
    Returns:
        True if texts are semantically similar
    """
    checker = get_default_checker()
    return checker.are_similar(text1, text2, threshold)

def semantic_similarity_score(text1: str, text2: str) -> float:
    """
    Convenience function to get semantic similarity score between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    checker = get_default_checker()
    return checker.calculate_similarity(text1, text2)