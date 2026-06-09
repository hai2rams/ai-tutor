"""Explicit ChromaDB seeder for MOE curriculum and reading passage vectors."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
READING_PASSAGES_PATH = DATA_DIR / "reading_passages.md"
CURRICULUM_PATH = DATA_DIR / "moe_curriculum.json"
VECTOR_STORE_PATH = DATA_DIR / "vector_store"
COLLECTION_NAME = "moe_knowledge_base"
TENANT_ID = "school-abc"
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
SNIPPET_LENGTH = 120


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str]


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma embedding function backed by the native Google Gemini API."""

    def __init__(self, api_key: str, model_name: str = GEMINI_EMBEDDING_MODEL) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []

        response = self._client.models.embed_content(
            model=self._model_name,
            contents=list(input),
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        if not response.embeddings:
            raise RuntimeError("Gemini embed_content returned no embeddings.")

        return [list(embedding.values) for embedding in response.embeddings]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "section"


def chunk_reading_passages(raw_text: str) -> list[TextChunk]:
    """Split markdown content into logical ## sections."""
    sections = [section.strip() for section in re.split(r"\n##\s+", raw_text) if section.strip()]
    chunks: list[TextChunk] = []

    for index, section in enumerate(sections, start=1):
        lines = section.splitlines()
        heading = lines[0].strip() if lines else f"section_{index}"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        text = f"{heading}\n\n{body}".strip()

        chunks.append(
            TextChunk(
                chunk_id=f"reading_passages_{_slugify(heading)}",
                text=text,
                metadata={"tenant_id": TENANT_ID, "source": "reading_passages"},
            )
        )

    return chunks


def chunk_curriculum(curriculum: dict) -> list[TextChunk]:
    """Split curriculum JSON into one chunk per level/topic chapter."""
    chunks: list[TextChunk] = []

    for level, subjects in curriculum.items():
        if not isinstance(subjects, dict):
            continue

        for topic, details in subjects.items():
            if not isinstance(details, dict):
                continue

            objectives = details.get("learning_objectives", [])
            prerequisites = details.get("prerequisites", [])
            misconceptions = details.get("common_misconceptions", [])

            text = (
                f"Level: {level}\n"
                f"Topic: {topic}\n\n"
                f"Learning objectives:\n"
                + "\n".join(f"- {item}" for item in objectives)
                + "\n\nPrerequisites:\n"
                + "\n".join(f"- {item}" for item in prerequisites)
                + "\n\nCommon misconceptions:\n"
                + "\n".join(f"- {item}" for item in misconceptions)
            ).strip()

            chunks.append(
                TextChunk(
                    chunk_id=f"curriculum_{_slugify(level)}_{_slugify(topic)}",
                    text=text,
                    metadata={"tenant_id": TENANT_ID, "source": "curriculum"},
                )
            )

    return chunks


def load_chunks() -> list[TextChunk]:
    """Read static source files and return all logical chunks."""
    chunks: list[TextChunk] = []

    if not READING_PASSAGES_PATH.exists():
        raise FileNotFoundError(f"Missing reading passages file: {READING_PASSAGES_PATH}")

    reading_text = READING_PASSAGES_PATH.read_text(encoding="utf-8").strip()
    if reading_text:
        chunks.extend(chunk_reading_passages(reading_text))

    if not CURRICULUM_PATH.exists():
        raise FileNotFoundError(f"Missing curriculum file: {CURRICULUM_PATH}")

    curriculum = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8") or "{}")
    chunks.extend(chunk_curriculum(curriculum))

    if not chunks:
        raise ValueError("No chunks were produced from the static source files.")

    return chunks


def seed_collection(chunks: list[TextChunk], embedding_function: GeminiEmbeddingFunction) -> chromadb.Collection:
    """Create or reset the MOE knowledge base collection and insert chunks."""
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))

    existing = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"tenant_id": TENANT_ID, "description": "MOE curriculum and reading passages"},
    )

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )

    return collection


def inspect_collection(collection: chromadb.Collection) -> None:
    """Print vector count and a text snippet for every stored chunk."""
    total = collection.count()
    print(f"\nTotal vectors stored: {total}")

    stored = collection.get(include=["documents", "metadatas"])
    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    print("\nStored vectors:")
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        snippet = (document or "").replace("\n", " ")[:SNIPPET_LENGTH]
        print(f"- ID: {chunk_id}")
        print(f"  metadata: {metadata}")
        print(f"  snippet: {snippet}{'...' if len(document or '') > SNIPPET_LENGTH else ''}")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("your-gemini-"):
        raise RuntimeError("GEMINI_API_KEY is not configured. Set it in .env before seeding vectors.")

    chunks = load_chunks()
    print(f"Prepared {len(chunks)} logical chunks from static source files.")

    embedding_function = GeminiEmbeddingFunction(api_key=api_key)
    collection = seed_collection(chunks, embedding_function)

    print(f"Seeded collection '{COLLECTION_NAME}' at '{VECTOR_STORE_PATH}'.")
    inspect_collection(collection)


if __name__ == "__main__":
    main()
