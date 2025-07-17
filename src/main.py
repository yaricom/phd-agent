#!/usr/bin/env python3
"""
Multi-Agent Research System - Main Application

This is the main entry point for the PhD Agent multi-agent research system.
It provides a command-line interface for running research workflows.
"""

import argparse
import sys
import logging
from pathlib import Path

from phd_agent.agents.supervisor_agent import SupervisorAgent
from phd_agent.config import config

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System for Deep Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic research without PDFs
  python main.py --topic "Artificial Intelligence in Healthcare" --requirements "Analyze current applications and future trends"
  
  # Research with PDF documents only (no web search)
  python main.py --topic "Machine Learning" --requirements "Review recent advances" --pdfs papers/ --no-web-search
  
  # Research with PDF documents and web search
  python main.py --topic "Machine Learning" --requirements "Review recent advances" --pdfs papers/ research/
  
  # Custom configuration
  python main.py --topic "Climate Change" --requirements "Economic impacts" --max-sources 15 --essay-length long
        """,
    )

    parser.add_argument("--topic", required=True, help="Research topic to investigate")

    parser.add_argument(
        "--requirements",
        required=True,
        help="Research requirements and specific questions to address",
    )

    parser.add_argument(
        "--pdfs", nargs="*", help="Paths to PDF files or directories to process"
    )

    parser.add_argument(
        "--max-sources",
        type=int,
        default=10,
        help="Maximum number of sources to use (default: 10)",
    )

    parser.add_argument(
        "--essay-length",
        choices=["short", "medium", "long"],
        default="medium",
        help="Desired essay length (default: medium)",
    )

    parser.add_argument(
        "--output", help="Output file path for the essay (default: essay_output.txt)"
    )

    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Show only workflow status without running the full research",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search (use only PDF documents)",
    )

    args = parser.parse_args()

    # Apply command line overrides to configuration
    if args.no_web_search:
        config.ENABLE_WEB_SEARCH = False
        logger.info("Web search disabled via command line argument")

    # Validate PDF paths if provided
    pdf_paths = []
    if args.pdfs:
        for path in args.pdfs:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning(f"Path does not exist: {path}")
                continue
            pdf_paths.append(str(path_obj.absolute()))

    try:
        # Initialize supervisor agent
        logger.info("Initializing Multi-Agent Research System...")
        supervisor = SupervisorAgent()

        if args.status_only:
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
        logger.info(f"Starting research on: {args.topic}")
        logger.info(f"Requirements: {args.requirements}")
        logger.info(f"Max sources: {args.max_sources}")
        logger.info(f"Essay length: {args.essay_length}")
        logger.info(
            f"Web search: {'Enabled' if config.ENABLE_WEB_SEARCH else 'Disabled'}"
        )
        if pdf_paths:
            logger.info(f"PDF paths: {pdf_paths}")
        logger.info("-" * 50)

        state = supervisor.run(
            topic=args.topic,
            requirements=args.requirements,
            max_sources=args.max_sources,
            essay_length=args.essay_length,
            pdf_paths=pdf_paths,
        )

        # Display results
        logger.info("\n" + "=" * 50)
        logger.info("RESEARCH RESULTS")
        logger.info("=" * 50)

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
            from phd_agent.file_utils import write_essay, get_supported_formats

            output_file = args.output or "essay_output.txt"
            if write_essay(state.final_essay, output_file):
                logger.info(f"Essay saved to: {output_file}")
                logger.info(f"Supported formats: {', '.join(get_supported_formats())}")
            else:
                logger.error(f"Failed to save essay to: {output_file}")

            # Show essay content if verbose
            if args.verbose:
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

        logger.info("Research workflow completed!")

    except KeyboardInterrupt:
        logger.info("Research interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
