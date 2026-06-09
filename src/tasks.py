"""CrewAI task definitions and tutoring session orchestration."""

from __future__ import annotations

import os

from crewai import Crew, Process, Task

from src.agents import TutorAgents, build_agents
from src.tools import build_tenant_tools
from utils.db_sync import ensure_tenant_knowledge_base


class TutorSessionError(Exception):
    """Raised when the tutoring crew cannot produce a response."""


def _build_tasks(agents: TutorAgents) -> tuple[Task, Task, Task]:
    research_task = Task(
        description=(
            "Prepare curriculum context for tenant '{tenant_id}' and topic '{topic}'.\n"
            "Student ID: {student_id}\n"
            "Student question: {question}\n\n"
            "Use curriculum_lookup, reading_passage_search, and tenant_knowledge_search "
            "to collect objectives, prerequisites, and supporting passages."
        ),
        expected_output=(
            "A concise curriculum briefing with learning objectives, prerequisites, "
            "and any relevant reading support."
        ),
        agent=agents.curriculum_specialist,
    )

    tutoring_task = Task(
        description=(
            "Teach the student about '{topic}'.\n"
            "Student question: {question}\n\n"
            "Use the curriculum briefing from the previous task.\n"
            "Write a friendly response with:\n"
            "1. A direct answer\n"
            "2. A short worked example\n"
            "3. One check-for-understanding question"
        ),
        expected_output="A draft tutoring response suitable for the student.",
        agent=agents.tutor,
        context=[research_task],
    )

    review_task = Task(
        description=(
            "Review the draft tutoring response for topic '{topic}'.\n"
            "Use misconception_check against the draft answer.\n"
            "Fix factual errors, remove confusing language, and return the final "
            "student-facing answer only."
        ),
        expected_output="Final polished tutoring answer ready to show the student.",
        agent=agents.reviewer,
        context=[tutoring_task],
    )

    return research_task, tutoring_task, review_task


def run_tutor_session(
    tenant_id: str,
    student_id: str,
    topic: str,
    question: str,
) -> str:
    """Execute the tutoring crew and return the final student-facing answer."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("your-gemini-"):
        raise TutorSessionError(
            "GEMINI_API_KEY is not configured. Set a valid key in .env before running sessions."
        )

    ensure_tenant_knowledge_base(tenant_id)
    tools = build_tenant_tools(tenant_id)
    agents = build_agents(tenant_id, tools)
    research_task, tutoring_task, review_task = _build_tasks(agents)

    crew = Crew(
        agents=[agents.curriculum_specialist, agents.tutor, agents.reviewer],
        tasks=[research_task, tutoring_task, review_task],
        process=Process.sequential,
        verbose=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
    )

    try:
        result = crew.kickoff(
            inputs={
                "tenant_id": tenant_id,
                "student_id": student_id,
                "topic": topic,
                "question": question,
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface crew/provider failures to API layer
        raise TutorSessionError(f"Tutoring crew failed: {exc}") from exc

    answer = (result.raw or str(result)).strip()
    if not answer:
        raise TutorSessionError("Tutoring crew returned an empty response.")

    return answer
