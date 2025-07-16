import os
from typing import Optional
from pydantic import BaseSettings, validator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('phd_agent.log')
    ]
)

logger = logging.getLogger(__name__)

class Config(BaseSettings):
    """Configuration settings for the PhD Agent system."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7
    
    # Milvus Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "research_documents"
    
    # Text Processing Configuration
    MAX_TOKENS_PER_CHUNK: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Analysis Configuration
    RELEVANCE_THRESHOLD: float = 0.6
    
    # Web Search Configuration
    ENABLE_WEB_SEARCH: bool = True
    MAX_SEARCH_RESULTS: int = 10
    SEARCH_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"
    
    @validator('OPENAI_API_KEY')
    def validate_openai_key(cls, v):
        if not v:
            logger.warning("OpenAI API key not set")
        return v
    
    def validate(self):
        """Validate configuration settings."""
        if not self.OPENAI_API_KEY:
            logger.error("OpenAI API key is required")
            raise ValueError("OpenAI API key is required")
        
        logger.info("Configuration validated successfully")
        return True

# Global config instance
config = Config() 