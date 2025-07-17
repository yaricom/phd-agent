#!/usr/bin/env python3
"""
Basic Research Example

This example demonstrates how to use the PhD Agent multi-agent research system
for a simple research task without PDF documents.
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


logger = logging.getLogger(__name__)


def main():
    """Run a basic research example."""
    # import dependencies
    from phd_agent.agents.supervisor_agent import SupervisorAgent
    from phd_agent.file_utils import write_essay

    logger.info("=" * 60)
    logger.info("PhD Agent - Basic Research Example")
    logger.info("=" * 60)

    # Initialize the supervisor agent
    logger.info("Initializing supervisor agent...")
    supervisor = SupervisorAgent()

    # Define research parameters
    topic = "Artificial Intelligence in Education"
    requirements = """
    Analyze the current applications of AI in educational settings, including:
    1. Personalized learning systems
    2. Automated grading and assessment
    3. Intelligent tutoring systems
    4. Challenges and limitations
    5. Future trends and opportunities
    """

    logger.info(f"\nResearch Topic: {topic}")
    logger.info(f"Requirements: {requirements.strip()}")
    logger.info("-" * 60)

    try:
        # Run the research workflow
        logger.info("Starting research workflow...")
        state = supervisor.run(
            topic=topic, requirements=requirements, max_sources=8, essay_length="medium"
        )

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("RESEARCH RESULTS")
        logger.info("=" * 60)

        # Show workflow status
        status = supervisor.get_workflow_status(state)
        logger.info(f"Task: {status.task.topic}")
        logger.info(f"Current Step: {status.current_step}")
        logger.info(f"Documents Collected: {status.documents_collected}")
        logger.info(f"Search Results: {status.search_results}")
        logger.info(f"Has Essay: {status.has_essay}")

        if status.errors:
            logger.error(f"Errors encountered: {len(status.errors)}")
            for error in status.errors:
                logger.error(f"  - {error}")

        # Show essay if available
        if state.final_essay:
            logger.info(f"Essay Title: {state.final_essay.title}")
            logger.info(f"Word Count: {state.final_essay.word_count}")
            logger.info(f"Sources Used: {len(state.final_essay.sources)}")

            # Save essay to file using file_utils
            output_file = "ai_education_essay.txt"
            if write_essay(state.final_essay, output_file):
                logger.info(f"Essay saved to: {output_file}")
            else:
                logger.error(f"Failed to save essay to: {output_file}")

            # Show essay content
            logger.info("\n" + "=" * 60)
            logger.info("ESSAY CONTENT")
            logger.info("=" * 60)
            logger.info(state.final_essay.content)

        # Show analysis results if available
        if state.analysis_results and state.analysis_results.data_summary:
            logger.info("Analysis Results:")
            summary = state.analysis_results.data_summary
            logger.info(f"  - Total documents: {summary.total_documents}")
            logger.info(f"  - Source distribution: {summary.source_distribution}")
            logger.info(f"  - Data coverage: {summary.data_coverage}")
        else:
            logger.info("No analysis results available - research incomplete")

        logger.info("Research workflow completed successfully!")

    except Exception as e:
        logger.error(f"Error during research: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
