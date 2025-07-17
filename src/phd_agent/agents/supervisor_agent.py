import uuid
import json
import logging

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import SecretStr
from ..models import ResearchTask, AgentState
from .pdf_agent import PDFAgent
from .web_search_agent import WebSearchAgent
from .analyst_agent import AnalystAgent
from .essay_writer_agent import EssayWriterAgent
from ..config import config
from ..vector_store import get_vector_store

logger = logging.getLogger(__name__)


def initialize_state(task: ResearchTask) -> AgentState:
    """Initialize the agent state for a new research task."""
    return AgentState(task=task, current_step="initialized")


def create_research_task(
    topic: str,
    requirements: str,
    max_sources: int = 10,
    essay_length: str = "medium",
) -> ResearchTask:
    """Create a new research task."""
    return ResearchTask(
        id=str(uuid.uuid4()),
        topic=topic,
        requirements=requirements,
        max_sources=max_sources,
        essay_length=essay_length,
    )


def _fallback_decision_logic(state: AgentState) -> Dict[str, Any]:
    """Fallback logic for determining the next step when LLM fails."""
    if state.current_step == "initialized":
        return {
            "next_step": "pdf_processing",
            "reasoning": "Starting with PDF processing",
            "should_continue": True,
            "recommendations": ["Process any available PDF documents"],
        }
    elif state.current_step == "pdf_completed":
        if config.ENABLE_WEB_SEARCH:
            return {
                "next_step": "web_searching",
                "reasoning": "Moving to web search for additional sources",
                "should_continue": True,
                "recommendations": ["Search for relevant web content"],
            }
        else:
            return {
                "next_step": "analyzing_data",
                "reasoning": "Web search disabled, moving directly to analysis",
                "should_continue": True,
                "recommendations": ["Analyze PDF documents only"],
            }
    elif state.current_step == "web_search_completed":
        return {
            "next_step": "analyzing_data",
            "reasoning": "Analyzing collected documents for relevance",
            "should_continue": True,
            "recommendations": ["Filter and rank documents"],
        }
    elif state.current_step == "analysis_completed":
        return {
            "next_step": "writing_essay",
            "reasoning": "Writing essay with analyzed data",
            "should_continue": True,
            "recommendations": ["Create essay outline and write essay"],
        }
    elif state.current_step == "essay_completed":
        return {
            "next_step": "completed",
            "reasoning": "Research workflow completed",
            "should_continue": False,
            "recommendations": ["Review final essay"],
        }
    else:
        return {
            "next_step": "completed",
            "reasoning": "Unknown state, marking as completed",
            "should_continue": False,
            "recommendations": ["Check for errors"],
        }


class SupervisorAgent:
    """Supervisor agent that orchestrates the entire research workflow."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
            api_key=SecretStr(config.OPENAI_API_KEY),
        )

        # Initialize all agents
        self.pdf_agent = PDFAgent()
        self.web_search_agent = WebSearchAgent()
        self.analyst_agent = AnalystAgent()
        self.essay_writer_agent = EssayWriterAgent()

        self.workflow_prompt = ChatPromptTemplate.from_template(
            """
        You are a research supervisor managing a multi-agent research system. Analyze the current state and determine the next steps.
        
        Current Research Task:
        Topic: {topic}
        Requirements: {requirements}
        Max Sources: {max_sources}
        Essay Length: {essay_length}
        
        System Configuration:
        Web Search Enabled: {web_search_enabled}
        
        Current State:
        Step: {current_step}
        Documents Collected: {doc_count}
        Errors: {errors}
        
        Available Steps:
        1. pdf_processing - Process PDF documents
        2. web_searching - Search for web content (only if enabled)
        3. analyzing_data - Analyze and filter documents
        4. writing_essay - Write the final essay
        5. completed - Research is complete
        
        Determine the next action based on the current state. Consider:
        - Whether enough documents have been collected
        - Whether documents have been analyzed for relevance
        - Whether the essay has been written
        - Any errors that need to be addressed
        - If web search is disabled, skip web_searching step
        
        Respond with JSON:
        {{
            "next_step": "step_name",
            "reasoning": "Explanation of why this step is next",
            "should_continue": true/false,
            "recommendations": ["List any recommendations"]
        }}
        """
        )

    def determine_next_step(self, state: AgentState) -> Dict[str, Any]:
        """Determine the next step in the workflow based on current state."""
        try:
            # Prepare the prompt
            messages = self.workflow_prompt.format_messages(
                topic=state.task.topic,
                requirements=state.task.requirements,
                max_sources=state.task.max_sources,
                essay_length=state.task.essay_length,
                web_search_enabled=config.ENABLE_WEB_SEARCH,
                current_step=state.current_step,
                doc_count=len(state.documents),
                errors="; ".join(state.errors) if state.errors else "None",
            )

            # Get decision from LLM
            response = self.llm.invoke(messages)

            # Parse JSON response
            try:
                # Handle both string and list response formats
                content = (
                    response.content
                    if isinstance(response.content, str)
                    else str(response.content)
                )
                decision = json.loads(content)
            except json.JSONDecodeError:
                # Fallback decision logic
                decision = _fallback_decision_logic(state)

            return decision

        except Exception as e:
            logger.error(f"Error determining next step: {e}")
            return _fallback_decision_logic(state)

    def execute_step(
        self, state: AgentState, step: str, pdf_paths: Optional[List[str]] = None
    ) -> AgentState:
        """Execute a specific step in the workflow."""
        try:
            if step == "pdf_processing":
                logger.info("Supervisor: Executing PDF processing step...")
                state = self.pdf_agent.run(state, pdf_paths)

            elif step == "web_searching":
                if config.ENABLE_WEB_SEARCH:
                    logger.info("Supervisor: Executing web search step...")
                    state = self.web_search_agent.run(state)
                else:
                    logger.info(
                        "Supervisor: Web search disabled, skipping web search step..."
                    )
                    state.current_step = "web_search_completed"

            elif step == "analyzing_data":
                logger.info("Supervisor: Executing data analysis step...")
                state = self.analyst_agent.run(state)

            elif step == "writing_essay":
                logger.info("Supervisor: Executing essay writing step...")
                state = self.essay_writer_agent.run(state)

            else:
                logger.warning(f"Supervisor: Unknown step '{step}'")
                state.errors.append(f"Unknown step: {step}")

        except Exception as e:
            error_msg = f"Error executing step '{step}': {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)

        return state

    def run_research_workflow(
        self,
        topic: str,
        requirements: str,
        max_sources: int = 10,
        essay_length: str = "medium",
        pdf_paths: Optional[List[str]] = None,
    ) -> AgentState:
        """Run the complete research workflow."""
        logger.info(f"Supervisor: Starting research workflow for topic: {topic}")

        # Check that vector sore is configured properly - we do this early to fail fast
        get_vector_store()

        # Create task and initialize state
        task = create_research_task(topic, requirements, max_sources, essay_length)
        state = initialize_state(task)

        # Main workflow loop
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Supervisor: Workflow iteration {iteration}")
            logger.info(f"Current step: {state.current_step}")
            logger.info(f"Documents collected: {len(state.documents)}")

            # Determine next step
            decision = self.determine_next_step(state)
            next_step = decision.get("next_step", "completed")
            should_continue = decision.get("should_continue", False)

            logger.info(f"Next step: {next_step}")
            logger.info(
                f"Reasoning: {decision.get('reasoning', 'No reasoning provided')}"
            )

            # Execute the step
            state = self.execute_step(state, next_step, pdf_paths)

            # Clear pdf_paths after first use
            pdf_paths = None

            # Check if we should continue
            if not should_continue or next_step == "completed":
                logger.info("Supervisor: Workflow completed")
                break

            # Check for too many errors
            if len(state.errors) > 5:
                logger.error("Supervisor: Too many errors, stopping workflow")
                break

        # Final status
        if state.final_essay:
            logger.info("Supervisor: Research completed successfully!")
            logger.info(f"Essay title: {state.final_essay.title}")
            logger.info(f"Essay word count: {state.final_essay.word_count}")
        else:
            logger.warning(
                "Supervisor: Research workflow did not complete successfully"
            )

        return state

    def run(
        self,
        topic: str,
        requirements: str,
        max_sources: int = 10,
        essay_length: str = "medium",
        pdf_paths: Optional[List[str]] = None,
    ) -> AgentState:
        """Main execution method for the supervisor agent."""
        try:
            # Validate configuration
            config.validate()

            # Run the complete workflow
            state = self.run_research_workflow(
                topic, requirements, max_sources, essay_length, pdf_paths
            )

            return state

        except Exception as e:
            logger.error(f"Supervisor Agent error: {str(e)}")
            # Return error state
            task = create_research_task(topic, requirements, max_sources, essay_length)
            state = initialize_state(task)
            state.errors.append(f"Supervisor error: {str(e)}")
            return state
