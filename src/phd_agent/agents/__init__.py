"""
PhD Agent - Multi-Agent Research System

This package contains the core agents for the research system:
- PDFAgent: Processes and analyzes PDF documents
- WebSearchAgent: Performs web searches and content extraction
- AnalystAgent: Assesses data relevance and quality
- EssayWriterAgent: Generates essays from research data
- SupervisorAgent: Orchestrates the entire workflow
"""

from .supervisor_agent import SupervisorAgent
from .pdf_agent import PDFAgent
from .web_search_agent import WebSearchAgent
from .analyst_agent import AnalystAgent
from .essay_writer_agent import EssayWriterAgent

__all__ = [
    "SupervisorAgent",
    "PDFAgent",
    "WebSearchAgent",
    "AnalystAgent",
    "EssayWriterAgent",
]

__version__ = "1.0.0"
__author__ = "PhD Agent Team"
__description__ = "Multi-Agent Research System for Deep Research"
