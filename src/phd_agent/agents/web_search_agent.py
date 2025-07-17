import requests
import uuid
import logging
from typing import List, Optional
from urllib.parse import urlparse
import time

from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..models import DocumentSource, DocumentType, SearchResult, AgentState
from ..vector_store import get_vector_store
from ..config import config

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """Agent responsible for performing web searches and extracting content from web pages."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
            api_key=SecretStr(config.OPENAI_API_KEY)
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.MAX_TOKENS_PER_CHUNK,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def search_web(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Perform web search using DuckDuckGo."""
        if max_results is None:
            max_results = config.MAX_SEARCH_RESULTS
        
        search_results = []
        
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                
                for result in results:
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        url=result.get('link', ''),
                        snippet=result.get('body', ''),
                        content=None  # Will be filled later if needed
                    )
                    search_results.append(search_result)
                
                logger.info(f"Web Search: Found {len(search_results)} results for query: {query}")
                
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
        
        return search_results
    
    def extract_web_content(self, url: str) -> Optional[str]:
        """Extract content from a web page."""
        try:
            # Add delay to be respectful to servers
            time.sleep(1)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text from body
            body = soup.find('body')
            if body:
                text = body.get_text()
            else:
                text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text if text else None
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def process_search_results(self, search_results: List[SearchResult], extract_content: bool = True) -> List[DocumentSource]:
        """Process search results and optionally extract full content."""
        documents = []
        
        for result in search_results:
            try:
                # Extract full content if requested
                content = result.snippet
                if extract_content and result.url:
                    full_content = self.extract_web_content(result.url)
                    if full_content:
                        content = full_content
                
                # Split content into chunks if it's too long
                if len(content) > config.MAX_TOKENS_PER_CHUNK:
                    chunks = self.text_splitter.split_text(content)
                else:
                    chunks = [content]
                
                # Create document sources for each chunk
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        doc_source = DocumentSource(
                            id=str(uuid.uuid4()),
                            title=f"{result.title} - Chunk {i+1}" if len(chunks) > 1 else result.title,
                            content=chunk.strip(),
                            source_type=DocumentType.WEB,
                            url=result.url,
                            metadata={
                                "search_query": "web_search",
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "domain": urlparse(result.url).netloc if result.url else None,
                                "relevance_score": result.relevance_score
                            }
                        )
                        documents.append(doc_source)
                
            except Exception as e:
                logger.error(f"Error processing search result {result.title}: {e}")
                continue
        
        return documents
    
    def store_web_documents(self, documents: List[DocumentSource]) -> List[str]:
        """Store web documents in the vector database."""
        stored_ids = []
        
        for document in documents:
            try:
                doc_id = get_vector_store().add_document(document)
                stored_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Error storing web document {document.title}: {e}")
                continue
        
        logger.info(f"Stored {len(stored_ids)} web documents in vector database")
        return stored_ids
    
    def search_relevant_web_content(self, topic: str, requirements: str, max_results: Optional[int] = None) -> List[DocumentSource]:
        """Search for relevant web content based on topic and requirements."""
        # Create search queries
        search_queries = [
            topic,
            f"{topic} {requirements}",
            f"{topic} research",
            f"{topic} analysis"
        ]
        
        all_documents = []
        
        for query in search_queries:
            try:
                # Perform web search
                search_results = self.search_web(query, max_results=max_results)
                
                # Process and extract content
                documents = self.process_search_results(search_results, extract_content=True)
                
                # Store documents
                if documents:
                    stored_ids = self.store_web_documents(documents)
                    all_documents.extend(documents)
                
                # Add delay between searches
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error searching for query '{query}': {e}")
                continue
        
        return all_documents
    
    def run(self, state: AgentState) -> AgentState:
        """Main execution method for the web search agent."""
        try:
            state.current_step = "web_searching"
            
            # Search for relevant web content
            web_documents = self.search_relevant_web_content(
                topic=state.task.topic,
                requirements=state.task.requirements,
                max_results=state.task.max_sources
            )
            
            # Add web documents to state
            state.documents.extend(web_documents)
            
            # Update search results for reference
            for doc in web_documents:
                if doc.url:
                    search_result = SearchResult(
                        title=doc.title,
                        url=doc.url,
                        snippet=doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                        content=doc.content
                    )
                    state.search_results.append(search_result)
            
            state.current_step = "web_search_completed"
            logger.info(f"Web Search Agent: Found and processed {len(web_documents)} web documents")
            
        except Exception as e:
            error_msg = f"Web Search Agent error: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state 