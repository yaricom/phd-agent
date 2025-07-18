from phd_agent.models import TaskDetails, EssaySummary, WorkflowStatus, AgentState


def create_workflow_status(state: AgentState) -> WorkflowStatus:
    """Get a summary of the current workflow status."""
    status_data = {
        "task": TaskDetails(
            topic=state.task.topic,
            requirements=state.task.requirements,
            max_relevant_sources=state.task.max_relevant_sources,
            essay_length=state.task.essay_length,
        ),
        "current_step": state.current_step,
        "documents_collected": len(state.documents),
        "search_results": len(state.search_results),
        "errors": state.errors,
        "has_outline": state.essay_outline is not None,
        "has_essay": state.final_essay is not None,
        "analysis_results": state.analysis_results,
    }

    if state.final_essay:
        status_data["essay_summary"] = EssaySummary(
            title=state.final_essay.title,
            word_count=state.final_essay.word_count,
            sources_used=len(state.final_essay.sources),
        )

    return WorkflowStatus(**status_data)
