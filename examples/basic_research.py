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

from phd_agent.agents.supervisor_agent import SupervisorAgent
from phd_agent.config import config

logger = logging.getLogger(__name__)

def main():
    """Run a basic research example."""
    
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
            topic=topic,
            requirements=requirements,
            max_sources=8,
            essay_length="medium"
        )
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("RESEARCH RESULTS")
        logger.info("=" * 60)
        
        # Show workflow status
        status = supervisor.get_workflow_status(state)
        logger.info(f"Task: {status['task']['topic']}")
        logger.info(f"Current Step: {status['current_step']}")
        logger.info(f"Documents Collected: {status['documents_collected']}")
        logger.info(f"Search Results: {status['search_results']}")
        logger.info(f"Has Essay: {status['has_essay']}")
        
        if status['errors']:
            logger.error(f"Errors encountered: {len(status['errors'])}")
            for error in status['errors']:
                logger.error(f"  - {error}")
        
        # Show essay if available
        if state.final_essay:
            logger.info(f"Essay Title: {state.final_essay.title}")
            logger.info(f"Word Count: {state.final_essay.word_count}")
            logger.info(f"Sources Used: {len(state.final_essay.sources)}")
            
            # Save essay to file
            output_file = "ai_education_essay.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {state.final_essay.title}\n")
                f.write(f"Word Count: {state.final_essay.word_count}\n")
                f.write(f"Sources: {len(state.final_essay.sources)}\n")
                f.write("=" * 50 + "\n\n")
                f.write(state.final_essay.content)
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("SOURCES:\n")
                for i, source in enumerate(state.final_essay.sources, 1):
                    f.write(f"{i}. {source.title} ({source.source_type.value})\n")
                    if source.url:
                        f.write(f"   URL: {source.url}\n")
                    f.write("\n")
            
            logger.info(f"Essay saved to: {output_file}")
            
            # Show essay content
            logger.info("\n" + "=" * 60)
            logger.info("ESSAY CONTENT")
            logger.info("=" * 60)
            logger.info(state.final_essay.content)
        
        # Show analysis results if available
        if state.analysis_results:
            logger.info(f"Analysis Results:")
            if 'data_summary' in state.analysis_results:
                summary = state.analysis_results['data_summary']
                logger.info(f"  - Total documents: {summary.get('total_documents', 0)}")
                logger.info(f"  - Source distribution: {summary.get('source_distribution', {})}")
                logger.info(f"  - Data coverage: {summary.get('data_coverage', 'unknown')}")
        
        logger.info("Research workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during research: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 