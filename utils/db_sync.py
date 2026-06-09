"""ChromaDB helpers and tenant-scoped knowledge-base seeding."""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", DATA_DIR / "chroma"))

_chroma_client: ClientAPI | None = None


def get_chroma_client() -> ClientAPI:
    """Return a process-wide persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return _chroma_client


def get_tenant_collection(tenant_id: str) -> Collection:
    """Get or create a tenant-isolated Chroma collection."""
    safe_tenant = tenant_id.replace("/", "_").replace(" ", "_")
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=f"tenant_{safe_tenant}",
        metadata={"tenant_id": tenant_id},
    )


def _load_seed_documents() -> list[tuple[str, str, dict[str, str]]]:
    """Load local curriculum files into (id, document, metadata) tuples."""
    seeds: list[tuple[str, str, dict[str, str]]] = []

    curriculum_path = DATA_DIR / "moe_curriculum.json"
    if curriculum_path.exists():
        curriculum = json.loads(curriculum_path.read_text(encoding="utf-8") or "{}")
        for level, subjects in curriculum.items():
            if not isinstance(subjects, dict):
                continue
            for topic, details in subjects.items():
                seeds.append(
                    (
                        f"curriculum:{level}:{topic}",
                        json.dumps({"level": level, "topic": topic, "details": details}),
                        {"source": "moe_curriculum.json", "topic": topic, "level": level},
                    )
                )

    passages_path = DATA_DIR / "reading_passages.md"
    if passages_path.exists():
        text = passages_path.read_text(encoding="utf-8").strip()
        if text:
            seeds.append(
                (
                    "reading_passages",
                    text,
                    {"source": "reading_passages.md", "topic": "reading"},
                )
            )

    misconceptions_path = DATA_DIR / "misconception_guide.md"
    if misconceptions_path.exists():
        text = misconceptions_path.read_text(encoding="utf-8").strip()
        if text:
            seeds.append(
                (
                    "misconception_guide",
                    text,
                    {"source": "misconception_guide.md", "topic": "misconceptions"},
                )
            )

    return seeds


def ensure_tenant_knowledge_base(tenant_id: str) -> Collection:
    """Seed tenant collection from local data files when empty."""
    collection = get_tenant_collection(tenant_id)
    if collection.count() > 0:
        return collection

    seeds = _load_seed_documents()
    if not seeds:
        return collection

    ids, documents, metadatas = zip(*[
        (doc_id, document, {**meta, "tenant_id": tenant_id})
        for doc_id, document, meta in seeds
    ])
    collection.add(ids=list(ids), documents=list(documents), metadatas=list(metadatas))
    return collection


def query_tenant_knowledge(tenant_id: str, query: str, limit: int = 4) -> list[str]:
    """Semantic search over the tenant knowledge base."""
    collection = ensure_tenant_knowledge_base(tenant_id)
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=min(limit, collection.count()))
    documents = results.get("documents") or [[]]
    return [doc for doc in documents[0] if doc]
