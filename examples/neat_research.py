#!/usr/bin/env python3
"""
The research example to study NEAT algorithm.
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

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
    """Run a NEAT research example."""

    # import dependencies
    from phd_agent.models import ResearchParameters
    from phd_agent.research_manager import run_research

    logger.info("=" * 60)
    logger.info("PhD Agent - NEAT fundamentals Research Example")
    logger.info("=" * 60)

    # Define research parameters
    topic = "The NEAT algorithm fundamentals"
    requirements = """
**Analyze the NEAT algorithm fundamentals, including:**
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

    parameters = ResearchParameters(
        topic=topic,
        requirements=requirements,
        max_relevant_sources=100,
        essay_length="medium",
        pdf_paths=get_pdf_paths(),
        output_files=get_output_files(),
        verbose=False,
    )
    run_research(parameters, status_only=False)


if __name__ == "__main__":
    main()
