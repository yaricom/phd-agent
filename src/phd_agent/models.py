from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ResearchStep(str, Enum):
    """Enumeration of possible research workflow steps."""

    INITIALIZED = "initialized"
    COMPLETED = "completed"
    PDF_PROCESSING = "pdf_processing"
    PDF_COMPLETED = "pdf_processing_completed"
    WEB_SEARCHING = "web_searching"
    WEB_SEARCH_COMPLETED = "web_searching_completed"
    ANALYZING_DATA = "analyzing_data"
    ANALYSIS_COMPLETED = "analyzing_completed"
    WRITING_ESSAY = "writing_essay"
    ESSAY_COMPLETED = "writing_essay_completed"


class DocumentType(str, Enum):
    PDF = "pdf"
    WEB = "web"
    ESSAY = "essay"


class DocumentSource(BaseModel):
    """Represents a document source with metadata."""

    id: Optional[str] = None
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
    max_relevant_sources: int = 10
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


class EssayValidationResult(BaseModel):
    """Results of essay validation against requirements."""

    meets_length: bool = True
    covers_topic: bool = True
    has_sources: bool = True
    issues: List[str] = Field(default_factory=list)
    word_count: int = 0
    expected_length_range: Optional[tuple[int, float]] = None
    topic_coverage_score: Optional[float] = None

    @property
    def overall_valid(self) -> bool:
        """Compute overall validation result."""
        return (
            self.meets_length
            and self.covers_topic
            and self.has_sources
            and not self.issues
        )


class DocumentRelevanceAssessment(BaseModel):
    """Assessment of document relevance."""

    document_id: str
    relevance_score: float  # 0.0 to 1.0
    reasoning: str
    key_points: List[str]
    confidence: float  # 0.0 to 1.0


class DocumentQualityAssessment(BaseModel):
    """Assessment of document quality and reliability."""

    document_id: str
    credibility_score: float  # 0.0 to 1.0
    information_quality: float  # 0.0 to 1.0
    currency_score: float  # 0.0 to 1.0
    overall_quality: str  # low, medium, high
    biases_limitations: List[str]
    recommendation: str  # include, exclude


class CollectedDataSummary(BaseModel):
    """Summary of collected research data."""

    total_documents: int
    source_distribution: Dict[str, int]
    average_content_length: float
    total_content_length: int
    research_topic: str
    data_coverage: str


class AnalysisResults(BaseModel):
    """Results from data analysis and document assessment."""

    data_summary: Optional[CollectedDataSummary] = None
    relevance_assessments: List[DocumentRelevanceAssessment] = Field(
        default_factory=list
    )
    filtered_documents: List[str] = Field(default_factory=list)  # Document IDs
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    coverage_score: Optional[float] = None
    confidence_score: Optional[float] = None


class AgentState(BaseModel):
    """State shared between all agents."""

    task: ResearchTask
    documents: List[DocumentSource] = Field(default_factory=list)
    search_results: List[SearchResult] = Field(default_factory=list)
    essay_outline: Optional[EssayOutline] = None
    final_essay: Optional[Essay] = None
    essay_validation_result: Optional[EssayValidationResult] = None
    analysis_results: AnalysisResults = Field(default_factory=AnalysisResults)
    current_step: ResearchStep = ResearchStep.INITIALIZED
    errors: List[str] = Field(default_factory=list)


class AgentMessage(BaseModel):
    """Message passed between agents."""

    from_agent: str
    to_agent: str
    content: Any
    message_type: Literal["task", "data", "result", "error", "control"]
    timestamp: datetime = Field(default_factory=datetime.now)


class TaskDetails(BaseModel):
    """Details of a research task."""

    topic: str
    requirements: str
    max_relevant_sources: int
    essay_length: str


class EssaySummary(BaseModel):
    """Details of a completed essay."""

    title: str
    word_count: int
    sources_used: int


class WorkflowStatus(BaseModel):
    """Status information for a research workflow."""

    task: TaskDetails
    current_step: str
    documents_collected: int
    search_results: int
    errors: List[str]
    has_outline: bool
    has_essay: bool
    analysis_results: Optional[AnalysisResults] = None
    essay_summary: Optional[EssaySummary] = None


class ResearchParameters(BaseModel):
    """Parameters for a research workflow."""

    topic: str
    requirements: str
    max_relevant_sources: int
    essay_length: str
    output_files: List[str]
    pdf_paths: Optional[List[str]] = None
    verbose: bool
