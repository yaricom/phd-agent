"""
Mock Vector Store for Testing

This module provides a mock implementation of the vector store for testing
when Milvus is not available.
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
import numpy as np

from .models import DocumentSource

logger = logging.getLogger(__name__)


class MockVectorStore:
    """Mock vector database for testing without Milvus."""

    def __init__(self):
        self.documents = {}
        self.embeddings = {}
        logger.info("Mock Vector Store initialized (Milvus not available)")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for text."""
        # Simple hash-based embedding for testing
        import hashlib

        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        embedding = [
            float(b) / 255.0 for b in hash_bytes[:16]
        ]  # 16-dimensional mock embedding
        # Pad to 384 dimensions
        embedding.extend([0.0] * (384 - len(embedding)))
        return embedding

    def add_document(self, document: DocumentSource) -> str:
        """Add a document to the mock vector store."""
        if not document.id:
            document.id = str(uuid.uuid4())

        # Generate mock embedding
        embedding = self._get_embedding(document.content)

        # Store document and embedding
        self.documents[document.id] = document
        self.embeddings[document.id] = embedding

        logger.info(f"Mock: Added document: {document.title}")
        return document.id

    def search_similar(
        self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[DocumentSource]:
        """Search for similar documents using mock similarity."""
        if not self.documents:
            return []

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Calculate similarities (mock)
        similarities = []
        for doc_id, doc_embedding in self.embeddings.items():
            # Simple cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k documents
        results = []
        for doc_id, _ in similarities[:top_k]:
            results.append(self.documents[doc_id])

        return results

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentSource]:
        """Retrieve a document by ID."""
        return self.documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            del self.embeddings[doc_id]
            return True
        return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "total_documents": len(self.documents),
            "collection_name": "mock_collection",
        }
