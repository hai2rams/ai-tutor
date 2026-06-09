"""CrewAI agent definitions for the multi-tenant AI tutor platform."""

from __future__ import annotations

import os
from dataclasses import dataclass

from crewai import Agent, LLM
from crewai.tools import BaseTool

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
GEMINI_LLM = LLM(model=GEMINI_MODEL)


@dataclass(frozen=True)
class TutorAgents:
    curriculum_specialist: Agent
    tutor: Agent
    reviewer: Agent


def build_agents(tenant_id: str, tools: list[BaseTool]) -> TutorAgents:
    """Construct role-specific agents for a single tutoring crew run."""
    curriculum_tools = [tool for tool in tools if tool.name in {"curriculum_lookup", "reading_passage_search", "tenant_knowledge_search"}]
    tutor_tools = [tool for tool in tools if tool.name in {"tenant_knowledge_search", "reading_passage_search"}]
    reviewer_tools = [tool for tool in tools if tool.name == "misconception_check"]

    curriculum_specialist = Agent(
        role="Curriculum Specialist",
        goal=(
            f"Align tutoring content with MOE curriculum standards for tenant '{tenant_id}' "
            "and gather grade-appropriate context before teaching."
        ),
        backstory=(
            "You are a Singapore MOE curriculum expert. You map student questions to "
            "syllabus objectives, prerequisites, and reading support materials."
        ),
        tools=curriculum_tools,
        verbose=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
        allow_delegation=False,
        llm=GEMINI_LLM,
    )

    tutor = Agent(
        role="AI Tutor",
        goal=(
            "Deliver clear, encouraging, age-appropriate explanations that directly answer "
            "the student's question while reinforcing conceptual understanding."
        ),
        backstory=(
            "You are a patient hybrid tutor who blends Socratic questioning with worked "
            "examples. You never shame the student and you keep language accessible."
        ),
        tools=tutor_tools,
        verbose=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
        allow_delegation=False,
        llm=GEMINI_LLM,
    )

    reviewer = Agent(
        role="Learning Safety Reviewer",
        goal=(
            "Detect misconceptions, factual errors, and curriculum misalignment in draft "
            "tutoring responses before they are shown to the student."
        ),
        backstory=(
            "You specialize in common student errors and MOE misconception patterns. "
            "You refine answers for accuracy, clarity, and pedagogical safety."
        ),
        tools=reviewer_tools,
        verbose=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
        allow_delegation=False,
        llm=GEMINI_LLM,
    )

    return TutorAgents(
        curriculum_specialist=curriculum_specialist,
        tutor=tutor,
        reviewer=reviewer,
    )
