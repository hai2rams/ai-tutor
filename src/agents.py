"""CrewAI agent definitions for the multi-tenant AI tutor platform."""

from __future__ import annotations

import os
from dataclasses import dataclass

from crewai import Agent, LLM
from crewai.mcp.config import MCPServerHTTP

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
GEMINI_LLM = LLM(model=GEMINI_MODEL)

# CrewAI string MCP refs only allow https:// URLs; localhost http requires MCPServerHTTP.
MCP_SERVER_ENDPOINT = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")


@dataclass(frozen=True)
class TutorAgents:
    curriculum_specialist: Agent
    tutor: Agent
    reviewer: Agent


def _tenant_mcps(tenant_id: str) -> list[MCPServerHTTP]:
    """Attach tenant context to MCP tool discovery via HTTP headers."""
    return [
        MCPServerHTTP(
            url=MCP_SERVER_ENDPOINT,
            headers={"X-Tenant-ID": tenant_id},
        )
    ]


def build_agents(tenant_id: str) -> TutorAgents:
    """Construct role-specific agents that discover tools from the MCP control plane."""
    mcps = _tenant_mcps(tenant_id)
    verbose = os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG"

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
        mcps=mcps,
        verbose=verbose,
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
        mcps=mcps,
        verbose=verbose,
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
        mcps=mcps,
        verbose=verbose,
        allow_delegation=False,
        llm=GEMINI_LLM,
    )

    return TutorAgents(
        curriculum_specialist=curriculum_specialist,
        tutor=tutor,
        reviewer=reviewer,
    )
