"""Custom CrewAI tools for curriculum lookup, misconceptions, and tenant RAG."""

from __future__ import annotations

import json
import re
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from utils.db_sync import DATA_DIR, query_tenant_knowledge

PROJECT_DATA = DATA_DIR


class TopicQuery(BaseModel):
    topic: str = Field(..., description="Curriculum topic or learning objective to look up.")


class MisconceptionQuery(BaseModel):
    topic: str = Field(..., description="Topic being taught.")
    draft_answer: str = Field(..., description="Draft tutoring response to validate.")


class RagQuery(BaseModel):
    query: str = Field(..., description="Natural-language search query for tenant knowledge base.")


class CurriculumLookupTool(BaseTool):
    name: str = "curriculum_lookup"
    description: str = (
        "Look up Singapore MOE curriculum standards, learning objectives, "
        "prerequisites, and common misconceptions for a topic."
    )
    args_schema: type[BaseModel] = TopicQuery
    tenant_id: str

    def _run(self, topic: str) -> str:
        curriculum_path = PROJECT_DATA / "moe_curriculum.json"
        if not curriculum_path.exists():
            return "No curriculum file found."

        curriculum = json.loads(curriculum_path.read_text(encoding="utf-8") or "{}")
        topic_key = topic.strip().lower()
        matches: list[dict] = []

        for level, subjects in curriculum.items():
            if not isinstance(subjects, dict):
                continue
            for subject_topic, details in subjects.items():
                if topic_key in subject_topic.lower() or topic_key in json.dumps(details).lower():
                    matches.append({"level": level, "topic": subject_topic, "details": details})

        if not matches:
            return f"No curriculum entry matched topic '{topic}'."

        return json.dumps(matches, indent=2)


class ReadingPassageTool(BaseTool):
    name: str = "reading_passage_search"
    description: str = "Search grade-appropriate reading passages related to a topic."
    args_schema: type[BaseModel] = TopicQuery
    tenant_id: str

    def _run(self, topic: str) -> str:
        path = PROJECT_DATA / "reading_passages.md"
        if not path.exists():
            return "No reading passages file found."

        text = path.read_text(encoding="utf-8")
        topic_key = topic.strip().lower()
        sections = [section.strip() for section in re.split(r"\n##\s+", text) if section.strip()]

        hits = [section for section in sections if topic_key in section.lower()]
        if not hits:
            return f"No reading passage section matched topic '{topic}'."

        return "\n\n---\n\n".join(hits[:2])


class MisconceptionGuideTool(BaseTool):
    name: str = "misconception_check"
    description: str = (
        "Check a draft tutoring answer against known student misconceptions "
        "for the topic and suggest corrections."
    )
    args_schema: type[BaseModel] = MisconceptionQuery
    tenant_id: str

    def _run(self, topic: str, draft_answer: str) -> str:
        path = PROJECT_DATA / "misconception_guide.md"
        if not path.exists():
            return "No misconception guide found."

        guide = path.read_text(encoding="utf-8")
        topic_key = topic.strip().lower()
        sections = [section.strip() for section in re.split(r"\n##\s+", guide) if section.strip()]
        relevant = [section for section in sections if topic_key in section.lower()]

        context = "\n\n".join(relevant) if relevant else guide
        return (
            f"Topic: {topic}\n\n"
            f"Misconception guide:\n{context}\n\n"
            f"Draft answer to review:\n{draft_answer}"
        )


class TenantRagTool(BaseTool):
    name: str = "tenant_knowledge_search"
    description: str = (
        "Semantic search across the tenant's indexed curriculum, passages, "
        "and misconception guides in ChromaDB."
    )
    args_schema: type[BaseModel] = RagQuery
    tenant_id: str

    def _run(self, query: str) -> str:
        hits = query_tenant_knowledge(self.tenant_id, query)
        if not hits:
            return "No tenant knowledge base results found."
        return "\n\n---\n\n".join(hits)


def build_tenant_tools(tenant_id: str) -> list[BaseTool]:
    """Create a tenant-scoped toolset for CrewAI agents."""
    return [
        CurriculumLookupTool(tenant_id=tenant_id),
        ReadingPassageTool(tenant_id=tenant_id),
        MisconceptionGuideTool(tenant_id=tenant_id),
        TenantRagTool(tenant_id=tenant_id),
    ]
