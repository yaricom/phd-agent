import uuid
import logging
from typing import List, Optional, Dict, Any
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from datetime import datetime

from langchain_openai import OpenAIEmbeddings

from .config import config
from .models import DocumentSource, DocumentType

logger = logging.getLogger(__name__)


vector_store = None


def get_vector_store():
    global vector_store
    if vector_store is None:
        try:
            vector_store = MilvusVectorStore()
        except Exception as e:
            logger.error(f"Milvus is not available ({e})", exc_info=True)
            raise
    return vector_store


class MilvusVectorStore:
    """Vector database service using Milvus for document storage and retrieval."""

    def __init__(self):
        self.collection_name = config.MILVUS_COLLECTION_NAME
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.dimension = 256  # text-embedding-3-large dimension
        self.collection = None
        _connect()
        self._setup_collection()

    def _setup_collection(self):
        """Setup the collection schema and create if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        else:
            # Define schema
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True
                ),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(
                    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
                ),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=30),
            ]

            schema = CollectionSchema(
                fields, description="Research documents collection"
            )
            self.collection = Collection(self.collection_name, schema)

            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            self.collection.create_index("embedding", index_params)
            logger.info(f"Created new collection: {self.collection_name}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        embedding = self.embedding_model.embed_query(text)
        return embedding

    def add_document(self, document: DocumentSource) -> str:
        """Add a document to the vector store."""
        if not document.id:
            document.id = str(uuid.uuid4())

        # Check if a collection is available
        if self.collection is None:
            raise Exception("Milvus collection not available")

        # Generate embedding
        embedding = self._get_embedding(document.content)

        # Truncate fields to fit Milvus schema limits
        truncated_title = _truncate_field(document.title, 500)
        truncated_content = _truncate_field(document.content, 65535)
        truncated_url = _truncate_field(document.url or "", 1000)
        truncated_file_path = _truncate_field(document.file_path or "", 500)
        truncated_metadata = _truncate_field(str(document.metadata), 2000)

        # Log if truncation occurred
        if len(document.title) > 500:
            logger.warning(
                f"Title truncated from {len(document.title)} to {len(truncated_title)} characters: {document.title[:100]}..."
            )
        if len(document.content) > 65535:
            logger.warning(
                f"Content truncated from {len(document.content)} to {len(truncated_content)} characters"
            )
        if document.url and len(document.url) > 1000:
            logger.warning(
                f"URL truncated from {len(document.url)} to {len(truncated_url)} characters"
            )

        # Prepare data
        data = [
            [document.id],
            [truncated_title],
            [truncated_content],
            [document.source_type.value],
            [truncated_url],
            [truncated_file_path],
            [embedding],
            [truncated_metadata],
            [document.created_at.isoformat()],
        ]

        # Insert data
        self.collection.insert(data)
        self.collection.flush()

        logger.info(f"Added document: {truncated_title}")
        return document.id

    def search_similar(
        self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[DocumentSource]:
        """Search for similar documents."""
        # Check if a collection is available
        if self.collection is None:
            raise Exception("Milvus collection not available")

        # Load a collection before searching
        self.collection.load()

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Prepare search parameters
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # Execute search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=str(filter_dict) if filter_dict else None,
            output_fields=[
                "id",
                "title",
                "content",
                "source_type",
                "url",
                "file_path",
                "metadata",
                "created_at",
            ],
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
                file_path=hit.entity.get("file_path")
                if hit.entity.get("file_path")
                else None,
                metadata=eval(hit.entity.get("metadata"))
                if hit.entity.get("metadata")
                else {},
                created_at=datetime.fromisoformat(hit.entity.get("created_at")),
            )
            documents.append(doc)

        return documents

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentSource]:
        """Retrieve a document by ID."""
        # Check if a collection is available
        if self.collection is None:
            raise Exception("Milvus collection not available")

        self.collection.load()

        results = self.collection.query(
            expr=f'id == "{doc_id}"',
            output_fields=[
                "id",
                "title",
                "content",
                "source_type",
                "url",
                "file_path",
                "metadata",
                "created_at",
            ],
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
                created_at=datetime.fromisoformat(result.get("created_at")),
            )

        return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        # Check if a collection is available
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
        # Check if a collection is available
        if self.collection is None:
            return {
                "total_documents": 0,
                "collection_name": self.collection_name,
                "status": "not_available",
            }

        stats = {
            "total_documents": self.collection.num_entities,
            "collection_name": self.collection_name,
        }
        return stats


def store_documents(documents: List[DocumentSource]) -> List[str]:
    """Store documents in the vector database."""
    stored_ids = []

    for document in documents:
        try:
            doc_id = get_vector_store().add_document(document)
            stored_ids.append(doc_id)
        except Exception as e:
            logger.error(
                f"Error storing document {document.title}, reason: {e}", exc_info=True
            )
            continue

    return stored_ids


def search_local_documents(query: str, top_k: int = 5) -> List[DocumentSource]:
    """Search for relevant documents in the local vector database."""
    try:
        documents = get_vector_store().search_similar(query, top_k=top_k)
        return documents
    except Exception as e:
        logger.error(f"Error searching local documents: {e}", exc_info=True)
        return []


def _truncate_field(text: str, max_length: int) -> str:
    """Truncate text to fit within the specified maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _connect():
    """Connect to Milvus server."""
    try:
        connections.connect(
            alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT
        )
        logger.info(f"Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    except Exception:
        logger.error(
            "Failed to connect to Milvus. Set MILVUS_HOST and MILVUS_PORT in .env to enable Milvus."
        )
        raise
