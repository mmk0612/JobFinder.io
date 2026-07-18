"""
src/agents/registry.py
----------------------
Central registry for all orchestrator-recognized agents.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.agents.application_tracker_agent import ApplicationTrackerAgent
from src.agents.ats_optimization_agent import AtsOptimizationAgent
from src.agents.career_coach_agent import CareerCoachAgent
from src.agents.company_research_agent import CompanyResearchAgent
from src.agents.interview_prep_agent import InterviewPrepAgent
from src.agents.job_collection_agent import JobCollectionAgent
from src.agents.resume_tailoring_agent import ResumeTailoringAgent
from src.agents.resume_analysis_agent import ResumeAnalysisAgent

ALL_AGENT_NAMES = [
    ResumeAnalysisAgent.name,
    JobCollectionAgent.name,
    ResumeTailoringAgent.name,
    AtsOptimizationAgent.name,
    CompanyResearchAgent.name,
    ApplicationTrackerAgent.name,
    InterviewPrepAgent.name,
    CareerCoachAgent.name,
]


def build_default_agent_registry() -> dict[str, BaseAgent]:
    return {
        ResumeAnalysisAgent.name: ResumeAnalysisAgent(),
        JobCollectionAgent.name: JobCollectionAgent(),
        ResumeTailoringAgent.name: ResumeTailoringAgent(),
        AtsOptimizationAgent.name: AtsOptimizationAgent(),
        CompanyResearchAgent.name: CompanyResearchAgent(),
        ApplicationTrackerAgent.name: ApplicationTrackerAgent(),
        InterviewPrepAgent.name: InterviewPrepAgent(),
        CareerCoachAgent.name: CareerCoachAgent(),
    }
