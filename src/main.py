#!/usr/bin/env python3
"""
Multi-Agent Research System - Main Application

This is the main entry point for the PhD Agent multi-agent research system.
It provides a command-line interface for running research workflows.
"""

import argparse
import logging
from pathlib import Path

from phd_agent.config import config
from phd_agent.models import ResearchParameters
from phd_agent.research_manager import run_research

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

    parameters = ResearchParameters(
        topic=args.topic,
        requirements=args.requirements,
        max_relevant_sources=args.max_sources,
        essay_length=args.essay_length,
        pdf_paths=pdf_paths,
        output_files=[args.output],
        verbose=args.verbose,
    )
    run_research(parameters, status_only=args.status_only)


if __name__ == "__main__":
    main()
