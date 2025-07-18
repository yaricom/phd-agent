import logging
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

# Get the project root directory (2 levels up from this file)
project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(logs_dir / "phd_agent.log")],
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
    MAX_LOCAL_SEARCH_RESULTS: int = 1000

    # Analysis Configuration
    RELEVANCE_THRESHOLD: float = 0.6

    # Web Search Configuration
    ENABLE_WEB_SEARCH: bool = True
    MAX_WEB_SEARCH_RESULTS: int = 10
    SEARCH_TIMEOUT: int = 30

    class Config:
        env_file = ".env"

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key(cls, value: str) -> str:
        if not value:
            raise ValueError("OpenAI API key not set")
        logger.info("OpenAI API key set")
        return value


# Global config instance
config = Config()
