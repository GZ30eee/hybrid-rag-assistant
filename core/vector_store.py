from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """Add vectors with associated metadata."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for top-k similar vectors with optional metadata filter."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the store."""
        pass

class FAISSVectorStore(VectorStore):
    """FAISS-based vector store."""
    
    def __init__(self, dimension: int):
        import faiss
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        self.index.add(vectors)
        self.metadata.extend(metadata)
    
    def search(self, query_vector: np.ndarray, k: int, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        D, I = self.index.search(query_vector, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                # Apply metadata filter if provided
                if filter_metadata:
                    for key, value in filter_metadata.items():
                        if meta.get(key) != value:
                            break
                    else:
                        results.append({"metadata": meta, "score": score})
                else:
                    results.append({"metadata": meta, "score": score})
        return results
    
    def clear(self) -> None:
        self.index.reset()
        self.metadata = []

class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store."""
    
    def __init__(self, collection_name: str = "rag_collection", host: str = "localhost", port: int = 6333):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        from qdrant_client.http import models
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # This should be set dynamically; we'll update later
                    distance=models.Distance.COSINE
                )
            )
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        from qdrant_client.http import models
        ids = list(range(len(vectors)))
        payload = metadata  # expects list of dicts
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors.tolist(),
                payloads=payload
            )
        )
    
    def search(self, query_vector: np.ndarray, k: int, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        from qdrant_client.http import models
        filter_condition = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
            filter_condition = models.Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=k,
            query_filter=filter_condition
        )
        return [{"metadata": hit.payload, "score": hit.score} for hit in results]
    
    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()

def get_vector_store(store_type: str = "faiss", dimension: int = 384) -> VectorStore:
    """Factory function to return the appropriate vector store."""
    if store_type == "faiss":
        return FAISSVectorStore(dimension)
    elif store_type == "qdrant":
        return QdrantVectorStore()
    else:
        raise ValueError(f"Unsupported store type: {store_type}")
