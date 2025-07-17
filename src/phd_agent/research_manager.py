import logging
import sys

from phd_agent.agents import SupervisorAgent
from phd_agent.agents.agent_utils import create_workflow_status
from phd_agent.config import config
from phd_agent.models import ResearchParameters

logger = logging.getLogger(__name__)


def run_research(parameters: ResearchParameters, status_only: bool = False):
    try:
        # Initialize supervisor agent
        logger.info("Initializing Multi-Agent Research System...")
        supervisor = SupervisorAgent()

        if status_only:
            # Just show system status
            logger.info("System Status:")
            logger.info(
                f"- OpenAI API Key: {'Configured' if config.OPENAI_API_KEY else 'Missing'}"
            )
            logger.info(f"- Milvus Host: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
            logger.info(f"- Model: {config.OPENAI_MODEL}")
            logger.info(
                f"- Web Search: {'Enabled' if config.ENABLE_WEB_SEARCH else 'Disabled'}"
            )
            return

        # Run the research workflow
        logger.info(f"Starting research on: {parameters.topic}")
        logger.info(f"Requirements: {parameters.requirements}")
        logger.info(f"Max sources: {parameters.max_sources}")
        logger.info(f"Essay length: {parameters.essay_length}")
        logger.info(
            f"Web search: {'Enabled' if config.ENABLE_WEB_SEARCH else 'Disabled'}"
        )
        if parameters.pdf_paths:
            logger.info(f"PDF paths: {parameters.pdf_paths}")
        logger.info("-" * 50)

        state = supervisor.run(
            topic=parameters.topic,
            requirements=parameters.requirements,
            max_sources=parameters.max_sources,
            essay_length=parameters.essay_length,
            pdf_paths=parameters.pdf_paths,
        )

        # Display results
        logger.info("\n" + "=" * 50)
        logger.info("RESEARCH RESULTS")
        logger.info("=" * 50)

        # Show workflow status
        status = create_workflow_status(state)
        logger.info(f"Task: {status.task.topic}")
        logger.info(f"Current Step: {status.current_step}")
        logger.info(f"Documents Collected: {status.documents_collected}")
        logger.info(f"Search Results: {status.search_results}")
        logger.info(f"Has Essay: {status.has_essay}")

        if status.errors:
            logger.error(f"Errors encountered: {len(status.errors)}")
            for error in status.errors:
                logger.error(f"  - {error}")

        # Show an essay if available
        if state.final_essay:
            logger.info(f"Essay Title: {state.final_essay.title}")
            logger.info(f"Word Count: {state.final_essay.word_count}")
            logger.info(f"Sources Used: {len(state.final_essay.sources)}")

            # Save essay to file using file_utils
            from phd_agent.file_utils import write_essay, get_supported_formats

            logger.info("Writing essay to file(s)...")
            logger.info(f"Supported formats: {', '.join(get_supported_formats())}")

            for output_file in parameters.output_files:
                if write_essay(state.final_essay, output_file):
                    logger.info(f"Essay saved to: {output_file}")
                else:
                    logger.error(f"Failed to save essay to: {output_file}")

            # Show essay content if verbose
            if parameters.verbose:
                logger.info("\n" + "=" * 50)
                logger.info("ESSAY CONTENT")
                logger.info("=" * 50)
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

        if not status.errors:
            logger.info("Research workflow completed successfully!")
        else:
            logger.warning(
                "Research workflow completed with errors! Essay may not be valid."
            )

    except KeyboardInterrupt:
        logger.info("Research interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
