"""Standalone MCP knowledge server for curriculum, passages, and ChromaDB RAG."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from utils.db_sync import DATA_DIR, ensure_tenant_knowledge_base, query_tenant_knowledge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "school-abc")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
MCP_PATH = os.getenv("MCP_PATH", "/mcp")

mcp = FastMCP(
    name="ai-tutor-knowledge-mcp",
    instructions=(
        "MOE curriculum lookup, reading passage search, misconception checks, "
        "and tenant-scoped ChromaDB semantic search for the AI tutor platform."
    ),
    host=MCP_HOST,
    port=MCP_PORT,
    streamable_http_path=MCP_PATH,
)


def _resolve_tenant_id(tenant_id: str | None) -> str:
    return (tenant_id or DEFAULT_TENANT_ID).strip() or DEFAULT_TENANT_ID


def _lookup_curriculum(topic: str) -> str:
    curriculum_path = DATA_DIR / "moe_curriculum.json"
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


def _search_reading_passages(topic: str) -> str:
    path = DATA_DIR / "reading_passages.md"
    if not path.exists():
        return "No reading passages file found."

    text = path.read_text(encoding="utf-8")
    topic_key = topic.strip().lower()
    sections = [section.strip() for section in re.split(r"\n##\s+", text) if section.strip()]
    hits = [section for section in sections if topic_key in section.lower()]

    if not hits:
        return f"No reading passage section matched topic '{topic}'."

    return "\n\n---\n\n".join(hits[:2])


def _check_misconceptions(topic: str, draft_answer: str) -> str:
    path = DATA_DIR / "misconception_guide.md"
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


@mcp.tool()
def curriculum_lookup(topic: str, tenant_id: str = DEFAULT_TENANT_ID) -> str:
    """Look up Singapore MOE curriculum standards for a topic."""
    _resolve_tenant_id(tenant_id)
    return _lookup_curriculum(topic)


@mcp.tool()
def reading_passage_search(topic: str, tenant_id: str = DEFAULT_TENANT_ID) -> str:
    """Search grade-appropriate reading passages related to a topic."""
    _resolve_tenant_id(tenant_id)
    return _search_reading_passages(topic)


@mcp.tool()
def misconception_check(topic: str, draft_answer: str, tenant_id: str = DEFAULT_TENANT_ID) -> str:
    """Check a draft tutoring answer against known student misconceptions."""
    _resolve_tenant_id(tenant_id)
    return _check_misconceptions(topic, draft_answer)


@mcp.tool()
def tenant_knowledge_search(query: str, tenant_id: str = DEFAULT_TENANT_ID) -> str:
    """Semantic search across the tenant ChromaDB knowledge base."""
    tenant = _resolve_tenant_id(tenant_id)
    ensure_tenant_knowledge_base(tenant)
    hits = query_tenant_knowledge(tenant, query)
    if not hits:
        return "No tenant knowledge base results found."
    return "\n\n---\n\n".join(hits)


def run_mcp_server() -> None:
    """Start the streamable HTTP MCP server (default: http://0.0.0.0:8001/mcp)."""
    load_dotenv(PROJECT_ROOT / ".env")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_mcp_server()
