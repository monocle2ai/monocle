from typing import Union
import numpy as np
from pydantic_core import CoreSchema, PydanticSerializationError, core_schema
from sentence_transformers import SentenceTransformer
from monocle_test_tools.comparer.base_comparer import BaseComparer
import logging
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Fast and good for semantic similarity
# Alternative models:
# "all-mpnet-base-v2" - Better quality but slower
# "paraphrase-MiniLM-L6-v2" - Good for paraphrase detection
# "sentence-transformers/all-roberta-large-v1" - Highest quality

SIMILARITY_THRESHOLD: float = 0.8

class SentenceComparer(BaseComparer):
    _test_transformer: SentenceTransformer
    
    def __init__(self, **data):
        super().__init__(**data)
        self._test_transformer = SentenceTransformer(model_name_or_path=MODEL_NAME)

    def compare(self, expected: Union[dict, str], actual: Union[dict, str]) -> bool:
        if expected == actual:
            return True
        if not isinstance(expected, str) or not isinstance(actual, str):
            raise ValueError("Both expected and actual must be strings for sentence comparison.")
        return self._compare_sentences(expected, actual)

    def _compare_sentences(self, expected: str, actual: str) -> bool:
        similarity: dict = self._calculate_similarity(expected, actual, SIMILARITY_THRESHOLD)
        return similarity['are_similar']

    def _calculate_similarity(self, expected: str, actual: str, similarity_threshold: float):
        """
        Calculate similarity between two sentences using semantic embeddings
        
        Args:
            sentence1 (str): First sentence
            sentence2 (str): Second sentence
            model_param: Loaded sentence transformer model (uses global model if None)
            similarity_threshold (float): Threshold for considering sentences similar (0-1)
        
        Returns:
            dict: Contains similarity score, whether sentences are similar, and embeddings
        """
        # Generate embeddings for both sentences
        sentences: list[str] = [expected, actual]
        embeddings = self._generate_semantic_embeddings(sentences)

        # Calculate cosine similarity (since embeddings are normalized)
        similarity_score = np.dot(embeddings[0], embeddings[1])
        
        # Determine if sentences are similar based on threshold
        are_similar = similarity_score >= similarity_threshold
        
        return {
            'similarity_score': float(similarity_score),
            'are_similar': are_similar,
            'threshold': similarity_threshold,
            'sentence1': expected,
            'sentence2': actual,
            'embeddings': embeddings
        }

    def _generate_semantic_embeddings(self, texts: list[str]):
        """Generate semantic embeddings using sentence transformers"""
        
        # Generate embeddings
        embeddings = self._test_transformer.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=True
        )
        
        return embeddings.astype(np.float32)

