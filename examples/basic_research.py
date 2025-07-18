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


# Define the output files for the final essay output
output_files = [
    "ai_education_essay.txt",
    "ai_education_essay.pdf",
    "ai_education_essay.docx",
]


def get_output_files():
    """Get the full paths to the output files."""
    return [str(project_root / "output" / output_file) for output_file in output_files]


def main():
    """Run a basic research example."""

    # import dependencies
    from phd_agent.models import ResearchParameters
    from phd_agent.research_manager import run_research

    logger.info("=" * 60)
    logger.info("PhD Agent - Basic Research Example")
    logger.info("=" * 60)

    # Define research parameters
    topic = "Artificial Intelligence in Education"
    requirements = """
**Analyze the current applications of AI in educational settings, including:**
1. Personalized learning systems
2. Automated grading and assessment
3. Intelligent tutoring systems
4. Challenges and limitations
5. Future trends and opportunities
"""

    parameters = ResearchParameters(
        topic=topic,
        requirements=requirements,
        max_relevant_sources=80,
        essay_length="medium",
        pdf_paths=None,
        output_files=get_output_files(),
        verbose=False,
    )
    run_research(parameters, status_only=False)


if __name__ == "__main__":
    main()
