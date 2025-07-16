import uuid
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from pydantic import SecretStr

from ..models import DocumentSource, RelevanceAssessment, AgentState, AnalysisResults, CollectedDataSummary
from ..config import config

logger = logging.getLogger(__name__)

class AnalystAgent:
    """Agent responsible for analyzing and assessing the relevance of acquired data."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
            api_key=SecretStr(config.OPENAI_API_KEY)
        )
        
        self.relevance_prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst. Analyze the following document for its relevance to the research topic.
        
        Research Topic: {topic}
        Research Requirements: {requirements}
        
        Document Title: {title}
        Document Content: {content}
        Document Source: {source_type}
        
        Please assess this document and provide:
        1. Relevance Score (0.0 to 1.0, where 1.0 is highly relevant)
        2. Detailed reasoning for the score
        3. Key points or insights from the document
        4. Confidence level in your assessment (0.0 to 1.0)
        
        Format your response as JSON:
        {{
            "relevance_score": 0.85,
            "reasoning": "This document directly addresses the research topic by...",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "confidence": 0.9
        }}
        """)
        
        self.quality_prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst. Evaluate the quality and reliability of the following document.
        
        Document Title: {title}
        Document Content: {content}
        Document Source: {source_type}
        Document URL: {url}
        
        Please assess the document quality and provide:
        1. Credibility Score (0.0 to 1.0)
        2. Information Quality Score (0.0 to 1.0)
        3. Currency/Recency Score (0.0 to 1.0)
        4. Overall Quality Assessment
        5. Potential biases or limitations
        
        Format your response as JSON:
        {{
            "credibility_score": 0.8,
            "information_quality": 0.7,
            "currency_score": 0.9,
            "overall_quality": "high",
            "biases_limitations": ["List any biases or limitations"],
            "recommendation": "include" or "exclude"
        }}
        """)
    
    def assess_document_relevance(self, document: DocumentSource, topic: str, requirements: str) -> RelevanceAssessment:
        """Assess the relevance of a single document."""
        try:
            # Prepare the prompt
            messages = self.relevance_prompt.format_messages(
                topic=topic,
                requirements=requirements,
                title=document.title,
                content=document.content[:2000],  # Limit content length
                source_type=document.source_type.value
            )
            
            # Get assessment from LLM
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            import json
            try:
                # Handle both string and list response formats
                content = response.content if isinstance(response.content, str) else str(response.content)
                assessment_data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                assessment_data = {
                    "relevance_score": 0.5,
                    "reasoning": "Unable to parse assessment",
                    "key_points": [],
                    "confidence": 0.5
                }
            
            # Create assessment object
            assessment = RelevanceAssessment(
                document_id=document.id,
                relevance_score=assessment_data.get("relevance_score", 0.5),
                reasoning=assessment_data.get("reasoning", "No reasoning provided"),
                key_points=assessment_data.get("key_points", []),
                confidence=assessment_data.get("confidence", 0.5)
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing document {document.title}: {e}")
            # Return default assessment
            return RelevanceAssessment(
                document_id=document.id,
                relevance_score=0.5,
                reasoning=f"Error during assessment: {str(e)}",
                key_points=[],
                confidence=0.0
            )
    
    def assess_document_quality(self, document: DocumentSource) -> Dict[str, Any]:
        """Assess the quality and reliability of a document."""
        try:
            # Prepare the prompt
            messages = self.quality_prompt.format_messages(
                title=document.title,
                content=document.content[:2000],  # Limit content length
                source_type=document.source_type.value,
                url=document.url or "N/A"
            )
            
            # Get assessment from LLM
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            import json
            try:
                # Handle both string and list response formats
                content = response.content if isinstance(response.content, str) else str(response.content)
                quality_data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                quality_data = {
                    "credibility_score": 0.5,
                    "information_quality": 0.5,
                    "currency_score": 0.5,
                    "overall_quality": "medium",
                    "biases_limitations": ["Unable to assess"],
                    "recommendation": "include"
                }
            
            return quality_data
            
        except Exception as e:
            logger.error(f"Error assessing quality of document {document.title}: {e}")
            return {
                "credibility_score": 0.5,
                "information_quality": 0.5,
                "currency_score": 0.5,
                "overall_quality": "medium",
                "biases_limitations": [f"Error during assessment: {str(e)}"],
                "recommendation": "include"
            }
    
    def filter_documents_by_relevance(self, documents: List[DocumentSource], topic: str, requirements: str, threshold: float = 0.6) -> tuple[List[DocumentSource], List[RelevanceAssessment]]:
        """Filter documents based on relevance threshold."""
        relevant_documents = []
        assessments = []
        
        for document in documents:
            assessment = self.assess_document_relevance(document, topic, requirements)
            assessments.append(assessment)
            
            if assessment.relevance_score >= threshold:
                relevant_documents.append(document)
                logger.info(f"Document '{document.title}' - Relevance: {assessment.relevance_score:.2f}")
            else:
                logger.info(f"Document '{document.title}' - Relevance: {assessment.relevance_score:.2f} (below threshold)")
        
        return relevant_documents, assessments
    
    def rank_documents_by_quality(self, documents: List[DocumentSource]) -> List[DocumentSource]:
        """Rank documents by quality assessment."""
        document_qualities = []
        
        for document in documents:
            quality = self.assess_document_quality(document)
            document_qualities.append((document, quality))

            logger.info(f"Document '{document.title}' - Quality: {quality}")
        
        # Sort by overall quality score (average of credibility, information quality, and currency)
        def quality_score(quality_data):
            return (quality_data["credibility_score"] + 
                   quality_data["information_quality"] + 
                   quality_data["currency_score"]) / 3
        
        sorted_documents = sorted(document_qualities, key=lambda x: quality_score(x[1]), reverse=True)
        
        return [doc for doc, _ in sorted_documents]
    
    def generate_data_summary(self, documents: List[DocumentSource], topic: str) -> CollectedDataSummary:
        """Generate a summary of the collected data."""
        if not documents:
            return CollectedDataSummary(
                total_documents=0,
                source_distribution={},
                average_content_length=0.0,
                total_content_length=0,
                research_topic=topic,
                data_coverage="none"
            )
        
        # Count documents by source type
        source_counts = {}
        for doc in documents:
            source_type = doc.source_type.value
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        # Calculate average content length
        total_length = sum(len(doc.content) for doc in documents)
        avg_length = total_length / len(documents) if documents else 0
        
        # Determine data coverage
        if len(documents) >= 10:
            coverage = "comprehensive"
        elif len(documents) >= 5:
            coverage = "moderate"
        else:
            coverage = "limited"
        
        return CollectedDataSummary(
            total_documents=len(documents),
            source_distribution=source_counts,
            average_content_length=avg_length,
            total_content_length=total_length,
            research_topic=topic,
            data_coverage=coverage
        )
    
    def run(self, state: AgentState) -> AgentState:
        """Main execution method for the analyst agent."""
        try:
            state.current_step = "analyzing_data"
            
            if not state.documents:
                logger.info("Analyst Agent: No documents to analyze")
                state.current_step = "analysis_completed"
                return state
            
            # Assess relevance of all documents
            logger.info(f"Analyst Agent: Assessing relevance of {len(state.documents)} documents...")
            relevant_docs, assessments = self.filter_documents_by_relevance(
                state.documents,
                state.task.topic,
                state.task.requirements,
                threshold=config.RELEVANCE_THRESHOLD
            )
            
            # Rank documents by quality
            logger.info(f"Analyst Agent: Ranking {len(relevant_docs)} relevant documents by quality...")
            ranked_docs = self.rank_documents_by_quality(relevant_docs)
            
            # Update state with filtered and ranked documents
            state.documents = ranked_docs[:state.task.max_sources]  # Limit to max sources
            
            # Generate data summary
            summary = self.generate_data_summary(state.documents, state.task.topic)
            state.analysis_results = AnalysisResults(
                data_summary=summary,
                relevance_assessments=assessments,
                filtered_documents=[doc.id for doc in state.documents],
                quality_metrics={"total_assessed": len(assessments), "relevant_found": len(state.documents)}
            )
            
            state.current_step = "analysis_completed"
            logger.info(f"Analyst Agent: Analysis completed. {len(state.documents)} high-quality documents selected.")
            
        except Exception as e:
            error_msg = f"Analyst Agent error: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state 