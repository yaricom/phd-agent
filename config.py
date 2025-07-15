import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the multi-agent system."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    
    # Milvus Configuration
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "research_documents")
    
    # Web Search Configuration
    DUCKDUCKGO_MAX_RESULTS: int = int(os.getenv("DUCKDUCKGO_MAX_RESULTS", "10"))
    
    # System Configuration
    MAX_TOKENS_PER_CHUNK: int = int(os.getenv("MAX_TOKENS_PER_CHUNK", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Vector Database Configuration
    VECTOR_DIMENSION: int = 1536  # OpenAI embedding dimension
    METRIC_TYPE: str = "COSINE"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        return True

# Global config instance
config = Config() 