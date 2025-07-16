from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    WEB = "web"
    ESSAY = "essay"

class DocumentSource(BaseModel):
    """Represents a document source with metadata."""
    id: str
    title: str
    content: str
    source_type: DocumentType
    url: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
class SearchResult(BaseModel):
    """Represents a web search result."""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    relevance_score: Optional[float] = None

class ResearchTask(BaseModel):
    """Represents a research task with requirements."""
    id: str
    topic: str
    requirements: str
    max_sources: int = 10
    essay_length: str = "medium"  # short, medium, long
    focus_areas: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

class EssayOutline(BaseModel):
    """Represents an essay outline structure."""
    title: str
    introduction: str
    main_points: List[str]
    conclusion: str
    sources: List[str]

class Essay(BaseModel):
    """Represents a completed essay."""
    id: str
    title: str
    content: str
    outline: EssayOutline
    sources: List[DocumentSource]
    word_count: int
    created_at: datetime = Field(default_factory=datetime.now)

class AgentState(BaseModel):
    """State shared between all agents."""
    task: ResearchTask
    documents: List[DocumentSource] = Field(default_factory=list)
    search_results: List[SearchResult] = Field(default_factory=list)
    essay_outline: Optional[EssayOutline] = None
    final_essay: Optional[Essay] = None
    analysis_results: Dict[str, Any] = Field(default_factory=dict)
    current_step: str = "initialized"
    errors: List[str] = Field(default_factory=list)
    
class AgentMessage(BaseModel):
    """Message passed between agents."""
    from_agent: str
    to_agent: str
    content: Any
    message_type: Literal["task", "data", "result", "error", "control"]
    timestamp: datetime = Field(default_factory=datetime.now)

class RelevanceAssessment(BaseModel):
    """Assessment of document relevance."""
    document_id: str
    relevance_score: float  # 0.0 to 1.0
    reasoning: str
    key_points: List[str]
    confidence: float  # 0.0 to 1.0 