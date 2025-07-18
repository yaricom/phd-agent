import logging
import os
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import config
from ..models import DocumentSource, DocumentType, AgentState, ResearchStep
from ..vector_store import (
    store_documents,
    search_local_documents,
    get_documents_by_file_path,
)

logger = logging.getLogger(__name__)


class PDFAgent:
    """Agent responsible for processing PDF documents and storing them in the vector database."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.MAX_TOKENS_PER_CHUNK,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def process_pdf_file(self, file_path: str) -> List[DocumentSource]:
        """Process a single PDF file and extract documents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # check if we already have this file in the vector store
        documents = get_documents_by_file_path(file_path)
        if documents:
            logger.info(
                f"PDF [{file_path}] already processed -> {len(documents)} chunks. Using saved data."
            )
            return documents

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
            title = metadata.get("title", None)
            if title is None:
                title = Path(file_path).stem

            # Split text into chunks
            chunks = self.text_splitter.split_text(full_text)

            # Create document sources for each chunk
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    doc_source = DocumentSource(
                        title=f"{title} - Chunk {i + 1}",
                        content=chunk.strip(),
                        source_type=DocumentType.PDF,
                        file_path=file_path,
                        metadata={
                            "page_range": f"Chunk {i + 1}",
                            "total_chunks": len(chunks),
                            "original_title": title,
                            "file_size": os.path.getsize(file_path),
                            "pdf_metadata": metadata,
                        },
                    )
                    documents.append(doc_source)

            doc.close()
            logger.info(f"Processed PDF: {file_path} -> {len(documents)} chunks")

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}", exc_info=True)
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

    def run(
        self, state: AgentState, pdf_paths: Optional[List[str]] = None
    ) -> AgentState:
        """Main execution method for the PDF agent."""
        try:
            state.current_step = ResearchStep.PDF_PROCESSING

            if pdf_paths:
                # Process new PDF files
                documents = []
                for pdf_path in pdf_paths:
                    if os.path.isfile(pdf_path):
                        docs = self.process_pdf_file(pdf_path)
                        documents.extend(docs)
                    elif os.path.isdir(pdf_path):
                        docs = self.process_pdf_directory(pdf_path)
                        documents.extend(docs)

                if len(documents) > 0:
                    # Store only new documents
                    new_documents = [doc for doc in documents if doc.id is None]
                    stored_ids = store_documents(new_documents)
                    assert len(stored_ids) == len(
                        new_documents
                    ), f"Failed to store new documents {len(stored_ids)} != {len(new_documents)}"

                    logger.info(
                        f"Processed {len(documents)} PDF documents and stored {len(stored_ids)} new PDF documents"
                    )

            # Search for relevant documents based on the task
            relevant_docs = search_local_documents(
                state.task.topic + " " + state.task.requirements,
                top_k=config.MAX_LOCAL_SEARCH_RESULTS,
            )

            # Add relevant documents to state
            state.documents.extend(relevant_docs)

            logger.info(
                f"Found {len(relevant_docs)} relevant documents from local storage"
            )

            state.current_step = ResearchStep.PDF_COMPLETED

        except Exception as e:
            error_msg = f"PDF processing error: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

        return state
