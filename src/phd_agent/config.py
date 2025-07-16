from pydantic import field_validator
from pydantic_settings import BaseSettings
import logging
import os
from pathlib import Path

# Get the project root directory (2 levels up from this file)
project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / "phd_agent.log")
    ]
)

logger = logging.getLogger(__name__)

class Config(BaseSettings):
    """Configuration settings for the PhD Agent system."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4.1-mini"
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
    
    @field_validator('OPENAI_API_KEY')
    @classmethod
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