#!/usr/bin/env python3
"""
Basic Research Example

This example demonstrates how to use the PhD Agent multi-agent research system
for a simple research task without PDF documents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.supervisor_agent import SupervisorAgent
from config import config

def main():
    """Run a basic research example."""
    
    print("=" * 60)
    print("PhD Agent - Basic Research Example")
    print("=" * 60)
    
    # Initialize the supervisor agent
    print("Initializing supervisor agent...")
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
    
    print(f"\nResearch Topic: {topic}")
    print(f"Requirements: {requirements.strip()}")
    print("-" * 60)
    
    try:
        # Run the research workflow
        print("Starting research workflow...")
        state = supervisor.run(
            topic=topic,
            requirements=requirements,
            max_sources=8,
            essay_length="medium"
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("RESEARCH RESULTS")
        print("=" * 60)
        
        # Show workflow status
        status = supervisor.get_workflow_status(state)
        print(f"Task: {status['task']['topic']}")
        print(f"Current Step: {status['current_step']}")
        print(f"Documents Collected: {status['documents_collected']}")
        print(f"Search Results: {status['search_results']}")
        print(f"Has Essay: {status['has_essay']}")
        
        if status['errors']:
            print(f"\nErrors encountered: {len(status['errors'])}")
            for error in status['errors']:
                print(f"  - {error}")
        
        # Show essay if available
        if state.final_essay:
            print(f"\nEssay Title: {state.final_essay.title}")
            print(f"Word Count: {state.final_essay.word_count}")
            print(f"Sources Used: {len(state.final_essay.sources)}")
            
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
            
            print(f"\nEssay saved to: {output_file}")
            
            # Show essay content
            print("\n" + "=" * 60)
            print("ESSAY CONTENT")
            print("=" * 60)
            print(state.final_essay.content)
        
        # Show analysis results if available
        if state.analysis_results:
            print(f"\nAnalysis Results:")
            if 'data_summary' in state.analysis_results:
                summary = state.analysis_results['data_summary']
                print(f"  - Total documents: {summary.get('total_documents', 0)}")
                print(f"  - Source distribution: {summary.get('source_distribution', {})}")
                print(f"  - Data coverage: {summary.get('data_coverage', 'unknown')}")
        
        print("\nResearch workflow completed successfully!")
        
    except Exception as e:
        print(f"\nError during research: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 