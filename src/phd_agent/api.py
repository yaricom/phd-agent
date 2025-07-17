"""
FastAPI Web Interface for Multi-Agent Research System

This module provides a REST API for the PhD Agent multi-agent research system.
"""

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from phd_agent.agents.agent_utils import create_workflow_status
from phd_agent.agents.supervisor_agent import (
    SupervisorAgent,
    create_research_task,
    initialize_state,
)
from phd_agent.config import config
from phd_agent.models import AgentState

# Initialize FastAPI app
app = FastAPI(
    title="PhD Agent - Multi-Agent Research System",
    description="A comprehensive multi-agent system for deep research using PDF analysis, web search, and AI-powered essay writing",
    version="1.0.0",
)

# Global supervisor instance
supervisor = SupervisorAgent()

# In-memory storage for research tasks (in production, use a database)
research_tasks: Dict[str, AgentState] = {}
pdf_paths_storage: Dict[str, List[str]] = {}


class ResearchRequest(BaseModel):
    """Request model for starting a research task."""

    topic: str
    requirements: str
    max_sources: int = 10
    essay_length: str = "medium"
    enable_web_search: bool = True


class ResearchResponse(BaseModel):
    """Response model for research task status."""

    task_id: str
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class TaskStatus(BaseModel):
    """Model for task status information."""

    task_id: str
    topic: str
    current_step: str
    documents_collected: int
    search_results: int
    has_essay: bool
    errors: List[str]
    created_at: str


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PhD Agent - Multi-Agent Research System",
        "version": "1.0.0",
        "endpoints": {
            "start_research": "/research/start",
            "get_status": "/research/{task_id}/status",
            "get_essay": "/research/{task_id}/essay",
            "list_tasks": "/research/tasks",
            "upload_pdfs": "/research/{task_id}/upload-pdfs",
        },
    }


@app.post("/research/start", response_model=ResearchResponse)
async def start_research(request: ResearchRequest):
    """Start a new research task."""
    try:
        # Create task ID
        task_id = str(uuid.uuid4())

        # Apply web search configuration
        if not request.enable_web_search:
            config.ENABLE_WEB_SEARCH = False

        # Create a research task
        task = create_research_task(
            topic=request.topic,
            requirements=request.requirements,
            max_sources=request.max_sources,
            essay_length=request.essay_length,
        )

        # Initialize state
        state = initialize_state(task)

        # Store in memory
        research_tasks[task_id] = state

        return ResearchResponse(
            task_id=task_id,
            status="started",
            message="Research task created successfully",
            data={
                "topic": request.topic,
                "requirements": request.requirements,
                "max_sources": request.max_sources,
                "essay_length": request.essay_length,
                "enable_web_search": request.enable_web_search,
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start research: {str(e)}"
        )


@app.get("/research/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a research task."""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    state = research_tasks[task_id]
    status = create_workflow_status(state)

    return TaskStatus(
        task_id=task_id,
        topic=status.task.topic,
        current_step=status.current_step,
        documents_collected=status.documents_collected,
        search_results=status.search_results,
        has_essay=status.has_essay,
        errors=status.errors,
        created_at=state.task.created_at.isoformat(),
    )


@app.post("/research/{task_id}/run")
async def run_research_task(task_id: str, background_tasks: BackgroundTasks):
    """Run a research task in the background."""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    # Add the research task to background tasks
    background_tasks.add_task(run_research_workflow, task_id)

    return {
        "task_id": task_id,
        "status": "running",
        "message": "Research task started in background",
    }


async def run_research_workflow(task_id: str):
    """Background task to run the research workflow."""
    try:
        state = research_tasks[task_id]

        # Run the workflow
        updated_state = supervisor.run_research_workflow(
            topic=state.task.topic,
            requirements=state.task.requirements,
            max_sources=state.task.max_sources,
            essay_length=state.task.essay_length,
        )

        # Update the stored state
        research_tasks[task_id] = updated_state

    except Exception as e:
        # Update state with error
        if task_id in research_tasks:
            research_tasks[task_id].errors.append(f"Workflow error: {str(e)}")


@app.get("/research/{task_id}/essay")
async def get_essay(task_id: str):
    """Get the final essay for a completed research task."""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    state = research_tasks[task_id]

    if not state.final_essay:
        raise HTTPException(status_code=404, detail="Essay not yet completed")

    return {
        "task_id": task_id,
        "essay": {
            "title": state.final_essay.title,
            "content": state.final_essay.content,
            "word_count": state.final_essay.word_count,
            "sources": [
                {
                    "title": source.title,
                    "type": source.source_type.value,
                    "url": source.url,
                }
                for source in state.final_essay.sources
            ],
            "outline": {
                "introduction": state.final_essay.outline.introduction,
                "main_points": state.final_essay.outline.main_points,
                "conclusion": state.final_essay.outline.conclusion,
            },
        },
    }


@app.get("/research/{task_id}/essay/download")
async def download_essay(task_id: str):
    """Download the essay as a text file."""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    state = research_tasks[task_id]

    if not state.final_essay:
        raise HTTPException(status_code=404, detail="Essay not yet completed")

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)

    try:
        # Write essay content
        temp_file.write(f"Title: {state.final_essay.title}\n")
        temp_file.write(f"Word Count: {state.final_essay.word_count}\n")
        temp_file.write(f"Sources: {len(state.final_essay.sources)}\n")
        temp_file.write("=" * 50 + "\n\n")
        temp_file.write(state.final_essay.content)
        temp_file.write("\n\n" + "=" * 50 + "\n")
        temp_file.write("SOURCES:\n")
        for i, source in enumerate(state.final_essay.sources, 1):
            temp_file.write(f"{i}. {source.title} ({source.source_type.value})\n")
            if source.url:
                temp_file.write(f"   URL: {source.url}\n")
            temp_file.write("\n")

        temp_file.close()

        return FileResponse(
            temp_file.name, media_type="text/plain", filename=f"essay_{task_id}.txt"
        )

    except Exception as e:
        # Clean up temp file
        Path(temp_file.name).unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to create download: {str(e)}"
        )


@app.post("/research/{task_id}/upload-pdfs")
async def upload_pdfs(task_id: str, files: List[UploadFile] = File(...)):
    """Upload PDF files for a research task."""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    # Create a temporary directory for PDFs
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []

    try:
        # Save uploaded files
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                continue

            file_path = Path(temp_dir) / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            pdf_paths.append(str(file_path))

        if not pdf_paths:
            raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

        # Store PDF paths for this task
        pdf_paths_storage[task_id] = pdf_paths

        return {
            "task_id": task_id,
            "status": "pdfs_uploaded",
            "message": f"Uploaded {len(pdf_paths)} PDF files",
            "pdf_files": [Path(p).name for p in pdf_paths],
        }

    except Exception as e:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload PDFs: {str(e)}")


@app.get("/research/tasks")
async def list_tasks():
    """List all research tasks."""
    tasks = []
    for task_id, state in research_tasks.items():
        tasks.append(
            {
                "task_id": task_id,
                "topic": state.task.topic,
                "current_step": state.current_step,
                "documents_collected": len(state.documents),
                "has_essay": state.final_essay is not None,
                "created_at": state.task.created_at.isoformat(),
            }
        )

    return {"tasks": tasks}


@app.delete("/research/{task_id}")
async def delete_task(task_id: str):
    """Delete a research task."""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    del research_tasks[task_id]

    return {
        "task_id": task_id,
        "status": "deleted",
        "message": "Task deleted successfully",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_configured": bool(config.OPENAI_API_KEY),
        "milvus_host": f"{config.MILVUS_HOST}:{config.MILVUS_PORT}",
        "model": config.OPENAI_MODEL,
        "web_search_enabled": config.ENABLE_WEB_SEARCH,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
