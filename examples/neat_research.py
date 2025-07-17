#!/usr/bin/env python3
"""
The research example to study NEAT algorithm.
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from phd_agent.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from phd_agent.file_utils import write_essay, get_supported_formats  # noqa: E402

logger = logging.getLogger(__name__)


# Define the output files for the final essay output
output_files = ["neat_essay.txt", "neat_essay.pdf", "neat_essay.docx"]
# Define the PDF files to be used for the research
pdf_files = ["Evolving NN through Augmenting Topologies.pdf"]


def get_pdf_paths():
    """Get the full paths to the PDF files."""
    return [str(project_root / "data" / pdf_file) for pdf_file in pdf_files]


def get_output_files():
    """Get the full paths to the output files."""
    return [str(project_root / "output" / output_file) for output_file in output_files]


def main():
    """Run a basic research example."""

    logger.info("=" * 60)
    logger.info("PhD Agent - NEAT fundamentals Research Example")
    logger.info("=" * 60)

    # Initialize the supervisor agent
    logger.info("Initializing supervisor agent...")
    supervisor = SupervisorAgent()

    # Define research parameters
    topic = "The NEAT algorithm fundamentals"
    requirements = """
    Analyze the NEAT algorithm fundamentals, including:
    1. The evolutionary algorithms basis
    2. The historical origins of the NEAT algorithm
    3. The key components of the NEAT algorithm
    4. The fitness function and how it works
    5. The mutation and crossover operators
    6. The role of the innovation number
    7. The role of the crossover probability
    8. The role of the mutation probability
    9. The role of the population size
    10. The role of the number of generations
    11. The role of the elitism
    12. How NEAT algorithm is used to evolve artificial neural networks
    13. How it compares to other evolutionary algorithms
    14. How it compares to deep learning algorithms
    15. How it is used in the field of artificial intelligence
    16. How it is used in the field of robotics
    17. Future trends and opportunities
    """

    logger.info(f"\nResearch Topic: {topic}")
    logger.info(f"Requirements: {requirements.strip()}")
    logger.info("-" * 60)

    try:
        # Run the research workflow
        logger.info("Starting research workflow...")
        state = supervisor.run(
            topic=topic,
            requirements=requirements,
            max_sources=10,
            essay_length="medium",
            pdf_paths=get_pdf_paths(),
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
            for output_file in get_output_files():
                if write_essay(state.final_essay, output_file):
                    logger.info(f"Essay saved to: {output_file}")
                    logger.info(
                        f"Supported formats: {', '.join(get_supported_formats())}"
                    )
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
