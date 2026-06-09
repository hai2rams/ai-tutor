"""ChromaDB helpers, tenant knowledge seeding, and student mastery state sync."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", DATA_DIR / "chroma"))
CURRICULUM_PATH = DATA_DIR / "moe_curriculum.json"
MASTERY_PATH = DATA_DIR / "student_mastery.json"

MASTERY_PASS_THRESHOLD = float(os.getenv("MASTERY_PASS_THRESHOLD", "70"))
TopicStatus = Literal["REMEDIAL", "MASTERED"]

_chroma_client: ClientAPI | None = None
_mastery_lock = threading.Lock()

REMEDIAL_SIGNALS = (
    "misconception",
    "common mistake",
    "common error",
    "let's review",
    "let us review",
    "try again",
    "not quite",
    "struggled",
    "difficulty",
    "remedial",
    "revisit",
    "careful",
    "watch out",
)

MASTERED_SIGNALS = (
    "great job",
    "well done",
    "excellent",
    "you've got it",
    "correct",
    "nicely done",
    "mastered",
)


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

    if CURRICULUM_PATH.exists():
        curriculum = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8") or "{}")
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


def _load_curriculum() -> dict[str, Any]:
    if not CURRICULUM_PATH.exists():
        return {}
    return json.loads(CURRICULUM_PATH.read_text(encoding="utf-8") or "{}")


def resolve_curriculum_topic(topic: str) -> dict[str, Any] | None:
    """Match a topic name against MOE curriculum entries (case-insensitive)."""
    topic_key = topic.strip().lower()
    curriculum = _load_curriculum()

    for level, subjects in curriculum.items():
        if not isinstance(subjects, dict):
            continue
        for curriculum_topic, details in subjects.items():
            if curriculum_topic.lower() == topic_key:
                return {
                    "level": level,
                    "topic": curriculum_topic,
                    "details": details if isinstance(details, dict) else {},
                }

    return None


def derive_session_score(question: str, response_text: str) -> float:
    """Derive a mock mastery score from the tutoring conversation."""
    combined = f"{question}\n{response_text}".lower()
    score = 75.0

    for signal in REMEDIAL_SIGNALS:
        if signal in combined:
            score -= 8.0

    for signal in MASTERED_SIGNALS:
        if signal in combined:
            score += 5.0

    if "?" in question and any(word in question.lower() for word in ("why", "how", "confused", "don't understand")):
        score -= 5.0

    return max(0.0, min(100.0, round(score, 1)))


def _topic_status(score: float) -> TopicStatus:
    return "MASTERED" if score >= MASTERY_PASS_THRESHOLD else "REMEDIAL"


def _detect_learning_gaps(
    topic: str,
    response_text: str,
    score: float,
    curriculum_match: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Flag concept misunderstandings using curriculum metadata and session signals."""
    if score >= MASTERY_PASS_THRESHOLD and not any(signal in response_text.lower() for signal in REMEDIAL_SIGNALS):
        return []

    gaps: list[dict[str, Any]] = []
    response_lower = response_text.lower()
    timestamp = datetime.now(timezone.utc).isoformat()

    if curriculum_match:
        details = curriculum_match.get("details", {})
        misconceptions = details.get("common_misconceptions", [])
        for concept in misconceptions:
            concept_text = str(concept)
            concept_lower = concept_text.lower()
            keywords = [word for word in re.findall(r"[a-z]{4,}", concept_lower) if word not in {"with", "that", "this", "without"}]
            if any(keyword in response_lower for keyword in keywords[:4]) or score < MASTERY_PASS_THRESHOLD:
                gaps.append(
                    {
                        "topic": curriculum_match["topic"],
                        "level": curriculum_match["level"],
                        "concept": concept_text,
                        "detected_at": timestamp,
                        "session_score": score,
                        "evidence": response_text[:500],
                    }
                )

    if not gaps and score < MASTERY_PASS_THRESHOLD:
        gaps.append(
            {
                "topic": curriculum_match["topic"] if curriculum_match else topic,
                "level": curriculum_match["level"] if curriculum_match else "unknown",
                "concept": f"Needs reinforcement on {topic}",
                "detected_at": timestamp,
                "session_score": score,
                "evidence": response_text[:500],
            }
        )

    return gaps


def _load_mastery_store() -> dict[str, Any]:
    if not MASTERY_PATH.exists():
        return {"tenants": {}}
    return json.loads(MASTERY_PATH.read_text(encoding="utf-8") or '{"tenants": {}}')


def _save_mastery_store(store: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = MASTERY_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    temp_path.replace(MASTERY_PATH)


def _append_unique_gaps(existing: list[dict[str, Any]], new_gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {(gap.get("topic"), gap.get("concept")) for gap in existing}
    merged = list(existing)

    for gap in new_gaps:
        key = (gap.get("topic"), gap.get("concept"))
        if key in seen:
            continue
        merged.append(gap)
        seen.add(key)

    return merged


def update_student_mastery(
    tenant_id: str,
    student_id: str,
    topic: str,
    score: float,
    response_text: str,
) -> dict[str, Any]:
    """
    Persist student mastery state to data/student_mastery.json.

    Validates the topic against moe_curriculum.json, sets REMEDIAL or MASTERED
    status from the session score, and appends flagged learning gaps.
    """
    curriculum_match = resolve_curriculum_topic(topic)
    status = _topic_status(score)
    learning_gaps = _detect_learning_gaps(topic, response_text, score, curriculum_match)
    timestamp = datetime.now(timezone.utc).isoformat()

    session_log = {
        "topic": curriculum_match["topic"] if curriculum_match else topic,
        "score": score,
        "status": status,
        "response_excerpt": response_text[:300],
        "logged_at": timestamp,
    }

    with _mastery_lock:
        store = _load_mastery_store()
        tenant_bucket = store.setdefault("tenants", {}).setdefault(tenant_id, {"students": {}})
        student_bucket = tenant_bucket["students"].setdefault(
            student_id,
            {"topics": {}, "tracked_learning_gaps": []},
        )

        topic_key = (curriculum_match["topic"] if curriculum_match else topic).lower()
        topic_record = student_bucket["topics"].get(topic_key, {})
        topic_record.update(
            {
                "topic": curriculum_match["topic"] if curriculum_match else topic,
                "level": curriculum_match["level"] if curriculum_match else None,
                "status": status,
                "last_score": score,
                "last_updated": timestamp,
                "curriculum_aligned": curriculum_match is not None,
                "sessions": [*topic_record.get("sessions", []), session_log],
            }
        )
        student_bucket["topics"][topic_key] = topic_record
        student_bucket["tracked_learning_gaps"] = _append_unique_gaps(
            student_bucket.get("tracked_learning_gaps", []),
            learning_gaps,
        )

        _save_mastery_store(store)

    return topic_record
