#!/usr/bin/env python3
"""
Multi-Agent Research System - Main Application

This is the main entry point for the PhD Agent multi-agent research system.
It provides a command-line interface for running research workflows.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from agents.supervisor_agent import SupervisorAgent
from config import config

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System for Deep Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic research without PDFs
  python main.py --topic "Artificial Intelligence in Healthcare" --requirements "Analyze current applications and future trends"
  
  # Research with PDF documents
  python main.py --topic "Machine Learning" --requirements "Review recent advances" --pdfs papers/ research/
  
  # Custom configuration
  python main.py --topic "Climate Change" --requirements "Economic impacts" --max-sources 15 --essay-length long
        """
    )
    
    parser.add_argument(
        "--topic", 
        required=True,
        help="Research topic to investigate"
    )
    
    parser.add_argument(
        "--requirements", 
        required=True,
        help="Research requirements and specific questions to address"
    )
    
    parser.add_argument(
        "--pdfs",
        nargs="*",
        help="Paths to PDF files or directories to process"
    )
    
    parser.add_argument(
        "--max-sources",
        type=int,
        default=10,
        help="Maximum number of sources to use (default: 10)"
    )
    
    parser.add_argument(
        "--essay-length",
        choices=["short", "medium", "long"],
        default="medium",
        help="Desired essay length (default: medium)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path for the essay (default: essay_output.txt)"
    )
    
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Show only workflow status without running the full research"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate PDF paths if provided
    pdf_paths = []
    if args.pdfs:
        for path in args.pdfs:
            path_obj = Path(path)
            if not path_obj.exists():
                print(f"Warning: Path does not exist: {path}")
                continue
            pdf_paths.append(str(path_obj.absolute()))
    
    try:
        # Initialize supervisor agent
        print("Initializing Multi-Agent Research System...")
        supervisor = SupervisorAgent()
        
        if args.status_only:
            # Just show system status
            print("System Status:")
            print(f"- OpenAI API Key: {'Configured' if config.OPENAI_API_KEY else 'Missing'}")
            print(f"- Milvus Host: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
            print(f"- Model: {config.OPENAI_MODEL}")
            return
        
        # Run the research workflow
        print(f"\nStarting research on: {args.topic}")
        print(f"Requirements: {args.requirements}")
        print(f"Max sources: {args.max_sources}")
        print(f"Essay length: {args.essay_length}")
        if pdf_paths:
            print(f"PDF paths: {pdf_paths}")
        print("-" * 50)
        
        state = supervisor.run(
            topic=args.topic,
            requirements=args.requirements,
            max_sources=args.max_sources,
            essay_length=args.essay_length,
            pdf_paths=pdf_paths
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("RESEARCH RESULTS")
        print("=" * 50)
        
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
            output_file = args.output or "essay_output.txt"
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
            
            # Show essay content if verbose
            if args.verbose:
                print("\n" + "=" * 50)
                print("ESSAY CONTENT")
                print("=" * 50)
                print(state.final_essay.content)
        
        # Show analysis results if available
        if state.analysis_results:
            print(f"\nAnalysis Results:")
            if 'data_summary' in state.analysis_results:
                summary = state.analysis_results['data_summary']
                print(f"  - Total documents: {summary.get('total_documents', 0)}")
                print(f"  - Source distribution: {summary.get('source_distribution', {})}")
                print(f"  - Data coverage: {summary.get('data_coverage', 'unknown')}")
        
        print("\nResearch workflow completed!")
        
    except KeyboardInterrupt:
        print("\nResearch interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 