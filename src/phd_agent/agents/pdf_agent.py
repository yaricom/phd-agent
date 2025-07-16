import os
import fitz  # PyMuPDF
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ..models import DocumentSource, DocumentType, AgentState
from ..vector_store import vector_store
from ..config import config

logger = logging.getLogger(__name__)

class PDFAgent:
    """Agent responsible for processing PDF documents and storing them in the vector database."""
    
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
    
    def process_pdf_file(self, file_path: str) -> List[DocumentSource]:
        """Process a single PDF file and extract documents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        documents = []
        
        try:
            # Open PDF
            doc = fitz.open(file_path)
            
            # Extract text from each page
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()  # type: ignore
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            # Extract metadata
            metadata = doc.metadata or {}
            title = metadata.get('title', Path(file_path).stem)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            # Create document sources for each chunk
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    doc_source = DocumentSource(
                        id=str(uuid.uuid4()),
                        title=f"{title} - Chunk {i+1}",
                        content=chunk.strip(),
                        source_type=DocumentType.PDF,
                        file_path=file_path,
                        metadata={
                            "page_range": f"Chunk {i+1}",
                            "total_chunks": len(chunks),
                            "original_title": title,
                            "file_size": os.path.getsize(file_path),
                            "pdf_metadata": metadata
                        }
                    )
                    documents.append(doc_source)
            
            doc.close()
            logger.info(f"Processed PDF: {file_path} -> {len(documents)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
        
        return documents
    
    def process_pdf_directory(self, directory_path: str) -> List[DocumentSource]:
        """Process all PDF files in a directory."""
        pdf_files = list(Path(directory_path).glob("**/*.pdf"))
        all_documents = []
        
        for pdf_file in pdf_files:
            try:
                documents = self.process_pdf_file(str(pdf_file))
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        return all_documents
    
    def store_documents(self, documents: List[DocumentSource]) -> List[str]:
        """Store documents in the vector database."""
        stored_ids = []
        
        for document in documents:
            try:
                doc_id = vector_store.add_document(document)
                stored_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Error storing document {document.title}: {e}")
                continue
        
        logger.info(f"Stored {len(stored_ids)} documents in vector database")
        return stored_ids
    
    def search_local_documents(self, query: str, top_k: int = 5) -> List[DocumentSource]:
        """Search for relevant documents in the local vector database."""
        try:
            documents = vector_store.search_similar(query, top_k=top_k)
            return documents
        except Exception as e:
            logger.error(f"Error searching local documents: {e}")
            return []
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored documents."""
        try:
            stats = vector_store.get_collection_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {"error": str(e)}
    
    def run(self, state: AgentState, pdf_paths: Optional[List[str]] = None) -> AgentState:
        """Main execution method for the PDF agent."""
        try:
            state.current_step = "pdf_processing"
            
            if pdf_paths:
                # Process new PDF files
                new_documents = []
                for pdf_path in pdf_paths:
                    if os.path.isfile(pdf_path):
                        documents = self.process_pdf_file(pdf_path)
                        new_documents.extend(documents)
                    elif os.path.isdir(pdf_path):
                        documents = self.process_pdf_directory(pdf_path)
                        new_documents.extend(documents)
                
                # Store new documents
                if new_documents:
                    stored_ids = self.store_documents(new_documents)
                    state.documents.extend(new_documents)
                    logger.info(f"PDF Agent: Processed and stored {len(new_documents)} new documents")
            
            # Search for relevant documents based on the task
            relevant_docs = self.search_local_documents(
                state.task.topic + " " + state.task.requirements,
                top_k=state.task.max_sources
            )
            
            # Add relevant documents to state
            for doc in relevant_docs:
                if doc not in state.documents:
                    state.documents.append(doc)
            
            state.current_step = "pdf_completed"
            logger.info(f"PDF Agent: Found {len(relevant_docs)} relevant documents from local storage")
            
        except Exception as e:
            error_msg = f"PDF Agent error: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state 