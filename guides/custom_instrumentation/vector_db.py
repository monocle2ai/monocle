import os
import numpy as np
from typing import List, Dict, Any
import requests
import json

class InMemoryVectorDB:
    def __init__(self, api_key: str = None):
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.embedding_endpoint = "https://api.openai.com/v1/embeddings"
        self.model = "text-embedding-ada-002"


    def _get_embedding(self, text: str) -> List[float]:
        """Get embeddings from OpenAI API using direct HTTP request"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Add the model name as a class field
        
        data = {
            "input": text,
            "model": self.model
        }
        
        response = requests.post(
            self.embedding_endpoint,
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
            
        return response.json()['data'][0]['embedding']

    def store_text(self, id: str, text: str, metadata: Dict[str, Any] = None) -> None:
        """Store text by converting it to a vector first"""
        vector = self._get_embedding(text)
        self.vectors[id] = np.array(vector)
        if metadata:
            self.metadata[id] = metadata
        self.metadata.setdefault(id, {})['text'] = text

    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using text query"""
        query_vector = self._get_embedding(query_text)
        return self.search(query_vector, top_k)

    def store(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Store a vector with optional metadata"""
        self.vectors[id] = np.array(vector)
        if metadata:
            self.metadata[id] = metadata
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors using cosine similarity"""
        if not self.vectors:
            return []
        
        query_vec = np.array(query_vector)
        similarities = {}
        
        for id, vec in self.vectors.items():
            similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            similarities[id] = similarity
        
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            {
                "id": id,
                "similarity": score,
                "metadata": self.metadata.get(id, {})
            }
            for id, score in sorted_results
        ]
