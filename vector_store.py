import uuid
import logging
from typing import List, Optional, Dict, Any
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

from config import config
from models import DocumentSource, DocumentType

logger = logging.getLogger(__name__)

class MilvusVectorStore:
    """Vector database service using Milvus for document storage and retrieval."""
    
    def __init__(self):
        self.collection_name = config.MILVUS_COLLECTION_NAME
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.collection = None
        self._connect()
        self._setup_collection()
    
    def _connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT
            )
            logger.info(f"Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_collection(self):
        """Setup the collection schema and create if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        else:
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=30)
            ]
            
            schema = CollectionSchema(fields, description="Research documents collection")
            self.collection = Collection(self.collection_name, schema)
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def add_document(self, document: DocumentSource) -> str:
        """Add a document to the vector store."""
        if not document.id:
            document.id = str(uuid.uuid4())
        
        # Check if collection is available
        if self.collection is None:
            raise Exception("Milvus collection not available")
        
        # Generate embedding
        embedding = self._get_embedding(document.content)
        
        # Prepare data
        data = [
            [document.id],
            [document.title],
            [document.content],
            [document.source_type.value],
            [document.url or ""],
            [document.file_path or ""],
            [embedding],
            [str(document.metadata)],
            [document.created_at.isoformat()]
        ]
        
        # Insert data
        self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"Added document: {document.title}")
        return document.id
    
    def search_similar(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[DocumentSource]:
        """Search for similar documents."""
        # Check if collection is available
        if self.collection is None:
            raise Exception("Milvus collection not available")

        # Load collection before searching
        self.collection.load()
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Prepare search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # Execute search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=str(filter_dict) if filter_dict else None,
            output_fields=["id", "title", "content", "source_type", "url", "file_path", "metadata", "created_at"]
        )
        
        # Convert results to DocumentSource objects
        documents = []
        for hit in results:  # type: ignore
            doc = DocumentSource(
                id=hit.entity.get("id"),
                title=hit.entity.get("title"),
                content=hit.entity.get("content"),
                source_type=DocumentType(hit.entity.get("source_type")),
                url=hit.entity.get("url") if hit.entity.get("url") else None,
                file_path=hit.entity.get("file_path") if hit.entity.get("file_path") else None,
                metadata=eval(hit.entity.get("metadata")) if hit.entity.get("metadata") else {},
                created_at=datetime.fromisoformat(hit.entity.get("created_at"))
            )
            documents.append(doc)
        
        return documents
    
    def get_document_by_id(self, doc_id: str) -> Optional[DocumentSource]:
        """Retrieve a document by ID."""
        # Check if collection is available
        if self.collection is None:
            raise Exception("Milvus collection not available")
        
        self.collection.load()
        
        results = self.collection.query(
            expr=f'id == "{doc_id}"',
            output_fields=["id", "title", "content", "source_type", "url", "file_path", "metadata", "created_at"]
        )
        
        if results:
            result = results[0]
            return DocumentSource(
                id=result.get("id"),
                title=result.get("title"),
                content=result.get("content"),
                source_type=DocumentType(result.get("source_type")),
                url=result.get("url") if result.get("url") else None,
                file_path=result.get("file_path") if result.get("file_path") else None,
                metadata=eval(result.get("metadata")) if result.get("metadata") else {},
                created_at=datetime.fromisoformat(result.get("created_at"))
            )
        
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        # Check if collection is available
        if self.collection is None:
            raise Exception("Milvus collection not available")
        
        try:
            self.collection.delete(f'id == "{doc_id}"')
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        # Check if collection is available
        if self.collection is None:
            return {
                "total_documents": 0,
                "collection_name": self.collection_name,
                "status": "not_available"
            }
        
        stats = {
            "total_documents": self.collection.num_entities,
            "collection_name": self.collection_name
        }
        return stats

# Global vector store instance
try:
    vector_store = MilvusVectorStore()
except Exception as e:
    logger.warning(f"Milvus not available ({e}), using mock vector store")
    from vector_store_mock import mock_vector_store
    vector_store = mock_vector_store 