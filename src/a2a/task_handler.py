"""Parse A2A task payloads and map them to internal crew execution."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.a2a.schemas import A2AArtifact, A2AMessage, A2APart, A2ATaskRequest, A2ATaskResponse, A2ATaskStatus


class A2ATaskParseError(ValueError):
    """Raised when an A2A task payload cannot be mapped to a tutoring session."""


def _is_user_role(role: str) -> bool:
    normalized = role.strip().lower()
    return normalized in {"user", "role_user"}


def parse_tutoring_session(payload: A2ATaskRequest) -> tuple[str, str, str]:
    """Extract student_id, topic, and question from an A2A task request."""
    student_id = (
        (payload.metadata or {}).get("student_id")
        or (payload.metadata or {}).get("studentId")
        or "student-unknown"
    )
    topic = (payload.metadata or {}).get("topic") or "general"
    question_parts: list[str] = []

    for message in payload.messages:
        if not _is_user_role(message.role):
            continue

        if message.metadata:
            student_id = message.metadata.get("student_id", message.metadata.get("studentId", student_id))
            topic = message.metadata.get("topic", topic)

        for part in message.parts:
            if part.data:
                student_id = str(part.data.get("student_id", part.data.get("studentId", student_id)))
                topic = str(part.data.get("topic", topic))
                if "question" in part.data:
                    question_parts.append(str(part.data["question"]))
            if part.text:
                question_parts.append(part.text.strip())

    question = question_parts[-1] if question_parts else ""
    if not question:
        raise A2ATaskParseError("A2A task must include at least one user message with text or question data.")

    return student_id, topic, question


def build_completed_task_response(
    payload: A2ATaskRequest,
    answer: str,
    *,
    tenant_id: str,
    session_score: float,
    mastery_status: str,
) -> A2ATaskResponse:
    """Build a completed A2A task response with tutoring artifacts."""
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    agent_message = A2AMessage(
        message_id=str(uuid.uuid4()),
        role="ROLE_AGENT",
        parts=[A2APart(text=answer, media_type="text/plain")],
        metadata={"tenant_id": tenant_id},
    )

    return A2ATaskResponse(
        id=payload.task_id,
        context_id=payload.context_id,
        status=A2ATaskStatus(state="TASK_STATE_COMPLETED", timestamp=timestamp, message=agent_message),
        artifacts=[
            A2AArtifact(
                artifact_id=f"{payload.task_id}-answer",
                name="tutor_answer",
                description="Final student-facing tutoring response.",
                parts=[A2APart(text=answer, media_type="text/plain")],
                metadata={
                    "tenant_id": tenant_id,
                    "session_score": session_score,
                    "mastery_status": mastery_status,
                },
            )
        ],
        history=[*payload.messages, agent_message],
        metadata={
            "tenant_id": tenant_id,
            "student_id": (payload.metadata or {}).get("student_id"),
            "topic": (payload.metadata or {}).get("topic"),
        },
    )


def build_failed_task_response(payload: A2ATaskRequest, error_message: str) -> A2ATaskResponse:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return A2ATaskResponse(
        id=payload.task_id,
        context_id=payload.context_id,
        status=A2ATaskStatus(
            state="TASK_STATE_FAILED",
            timestamp=timestamp,
            message=A2AMessage(
                message_id=str(uuid.uuid4()),
                role="ROLE_AGENT",
                parts=[A2APart(text=error_message, media_type="text/plain")],
            ),
        ),
        history=payload.messages,
    )
