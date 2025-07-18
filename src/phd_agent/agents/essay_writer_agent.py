import json
import uuid
import logging
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import SecretStr

from ..llm_utils import parse_llm_response
from ..models import (
    DocumentSource,
    EssayOutline,
    Essay,
    AgentState,
    EssayValidationResult,
    ResearchStep,
)
from ..config import config

logger = logging.getLogger(__name__)


class EssayWriterAgent:
    """Agent responsible for writing essays using collected research data."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
            api_key=SecretStr(config.OPENAI_API_KEY),
        )

        self.outline_prompt = ChatPromptTemplate.from_template(
            """
        You are an expert academic writer. Create a detailed essay outline based on the research topic and collected data.
        
        Research Topic: {topic}
        Research Requirements: {requirements}
        Essay Length: {essay_length}
        
        Available Sources:
        {sources_summary}
        
        ***Create a comprehensive essay outline with:***
        1. A compelling title
        2. An engaging introduction that sets up the topic
        3. 3-5 main points that address the research requirements
        4. A strong conclusion that synthesizes the findings
        
        ***Respond with JSON:***
        {{
            "title": "Essay Title",
            "introduction": "Introduction text...",
            "main_points": [
                "Main point 1: Description",
                "Main point 2: Description",
                "Main point 3: Description"
            ],
            "conclusion": "Conclusion text...",
            "sources": ["Source 1", "Source 2", "Source 3"]
        }}
        
        You should only return the JSON response and nothing else. Do not include any additional text or formatting.
        """
        )

        self.essay_prompt = ChatPromptTemplate.from_template(
            """
        You are an expert academic writer. Write a comprehensive essay based on the provided outline and research data.
        
        Essay Outline:
        Title: {title}
        Introduction: {introduction}
        Main Points: {main_points}
        Conclusion: {conclusion}
        
        Research Topic: {topic}
        Research Requirements: {requirements}
        Essay Length: {essay_length}
        
        ***Available Research Data:***
        {research_data}
        
        ***Instructions:***
        1. Write a well-structured academic essay
        2. Use the provided research data to support your arguments
        3. Include proper citations and references
        4. Ensure the essay meets the specified length requirements
        5. Maintain academic tone and style
        6. Synthesize information from multiple sources
        
        ***Write the complete essay:***
        """
        )

    def create_essay_outline(self, state: AgentState) -> EssayOutline:
        """Create an essay outline based on the research topic and collected data."""
        try:
            # Prepare source summary
            sources_summary = []
            for i, doc in enumerate(
                state.documents[: state.task.max_relevant_sources], 1
            ):
                source_info = f"{i}. {doc.title} ({doc.source_type.value})"
                if doc.url:
                    source_info += f" - {doc.url}"
                sources_summary.append(source_info)

            sources_text = "\n".join(sources_summary)

            # Prepare the prompt
            messages = self.outline_prompt.format_messages(
                topic=state.task.topic,
                requirements=state.task.requirements,
                essay_length=state.task.essay_length,
                sources_summary=sources_text,
            )

            # Get outline from LLM
            response = self.llm.invoke(messages)
            # Handle both string and list response formats
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            # Parse JSON response
            try:
                outline_data = parse_llm_response(content)
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON response: '{content}'", exc_info=True)
                # Fallback outline if JSON parsing fails
                outline_data = {
                    "title": f"Research on {state.task.topic}",
                    "introduction": f"This essay explores {state.task.topic} based on comprehensive research.",
                    "main_points": [
                        f"Overview of {state.task.topic}",
                        "Key findings and analysis",
                        "Implications and conclusions",
                    ],
                    "conclusion": f"Summary of findings on {state.task.topic}",
                    "sources": [doc.title for doc in state.documents[:5]],
                }

            # Create an outline object
            outline = EssayOutline(
                title=outline_data.get("title", f"Research on {state.task.topic}"),
                introduction=outline_data.get("introduction", ""),
                main_points=outline_data.get("main_points", []),
                conclusion=outline_data.get("conclusion", ""),
                sources=outline_data.get("sources", []),
            )

            return outline

        except Exception as e:
            logger.error(f"Error creating essay outline: {e}", exc_info=True)
            # Return basic outline
            return EssayOutline(
                title=f"Research on {state.task.topic}",
                introduction=f"This essay explores {state.task.topic} based on comprehensive research.",
                main_points=[
                    f"Overview of {state.task.topic}",
                    "Key findings and analysis",
                    "Implications and conclusions",
                ],
                conclusion=f"Summary of findings on {state.task.topic}",
                sources=[doc.title for doc in state.documents[:5]],
            )

    def write_essay(self, state: AgentState, outline: EssayOutline) -> Essay:
        """Write the complete essay based on the outline and research data."""
        try:
            # Prepare research data
            research_data = _prepare_research_data(state.documents)

            # Prepare the prompt
            messages = self.essay_prompt.format_messages(
                title=outline.title,
                introduction=outline.introduction,
                main_points="\n".join(outline.main_points),
                conclusion=outline.conclusion,
                topic=state.task.topic,
                requirements=state.task.requirements,
                essay_length=state.task.essay_length,
                research_data=research_data,
            )

            # Get essay from LLM
            response = self.llm.invoke(messages)
            # Handle both string and list response formats
            essay_content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )

            # Calculate word count
            word_count = len(essay_content.split())

            # Create an essay object
            essay = Essay(
                id=str(uuid.uuid4()),
                title=outline.title,
                content=essay_content,
                outline=outline,
                sources=state.documents,
                word_count=word_count,
            )

            return essay

        except Exception as e:
            logger.error(f"Error writing essay: {e}", exc_info=True)
            # Return a basic essay
            basic_content = f"""
            {outline.title}
            
            {outline.introduction}
            
            {" ".join(outline.main_points)}
            
            {outline.conclusion}
            """

            return Essay(
                id=str(uuid.uuid4()),
                title=outline.title,
                content=basic_content,
                outline=outline,
                sources=state.documents,
                word_count=len(basic_content.split()),
            )

    def run(self, state: AgentState) -> AgentState:
        """Main execution method for the essay writer agent."""
        try:
            state.current_step = ResearchStep.WRITING_ESSAY

            if not state.documents:
                logger.info(
                    "Essay Writer Agent: No documents available for essay writing"
                )
                state.current_step = ResearchStep.ESSAY_COMPLETED
                return state

            # Create essay outline
            logger.info("Essay Writer Agent: Creating essay outline...")
            outline = self.create_essay_outline(state)
            state.essay_outline = outline

            # Write the essay
            logger.info("Essay Writer Agent: Writing essay...")
            essay = self.write_essay(state, outline)
            state.final_essay = essay

            # Validate essay
            validation = _validate_essay_requirements(essay, state.task.requirements)

            state.current_step = ResearchStep.ESSAY_COMPLETED
            state.essay_validation_result = validation

            logger.info(
                f"Essay Writer Agent: Essay completed. Word count: {essay.word_count}"
            )
            logger.info(
                f"Essay Writer Agent: Validation - Overall valid: {validation.overall_valid}, "
                f"Length: {validation.meets_length}, Topic: {validation.covers_topic}, "
                f"Sources: {validation.has_sources}, Issues: {len(validation.issues)}"
            )
            if validation.issues:
                for issue in validation.issues:
                    logger.warning(f"Essay Writer Agent: Validation issue - {issue}")

        except Exception as e:
            error_msg = f"Essay Writer Agent error: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

        return state


def _validate_essay_requirements(
    essay: Essay, requirements: str
) -> EssayValidationResult:
    """Validate that the essay meets the specified requirements."""
    result = EssayValidationResult(
        word_count=essay.word_count,
    )

    # Check length requirements
    if "short" in requirements.lower():
        result.expected_length_range = (0, 1000)
        if essay.word_count > 1000:
            result.meets_length = False
            result.issues.append("Essay is longer than requested for short format")
    elif "long" in requirements.lower():
        result.expected_length_range = (2000, float("inf"))
        if essay.word_count < 2000:
            result.meets_length = False
            result.issues.append("Essay is shorter than requested for long format")
    else:  # medium
        result.expected_length_range = (1000, 2000)
        if essay.word_count < 1000 or essay.word_count > 2000:
            result.meets_length = False
            result.issues.append(
                "Essay length does not meet medium format requirements"
            )

    # Check topic coverage
    topic_words = [word.lower() for word in requirements.split() if len(word) > 3]
    content_lower = essay.content.lower()
    topic_coverage = sum(1 for word in topic_words if word in content_lower)
    result.topic_coverage_score = (
        topic_coverage / len(topic_words) if topic_words else 0.0
    )

    if result.topic_coverage_score < 0.5:
        result.covers_topic = False
        result.issues.append("Essay may not fully cover the research topic")

    return result


def _prepare_research_data(documents: List[DocumentSource]) -> str:
    """Prepare research data for the essay writing prompt."""
    research_data = []

    for i, doc in enumerate(documents[:15], 1):  # Limit to top 15 sources
        # Truncate content for prompt
        content_preview = (
            doc.content[:800] + "..." if len(doc.content) > 800 else doc.content
        )

        source_info = f"Source {i}: {doc.title}\n"
        source_info += f"Type: {doc.source_type.value}\n"
        if doc.url:
            source_info += f"URL: {doc.url}\n"
        source_info += f"Content: {content_preview}\n"
        source_info += "-" * 50 + "\n"

        research_data.append(source_info)

    return "\n".join(research_data)
