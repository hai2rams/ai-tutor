"""A2A protocol request/response models (HTTP+JSON binding)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class A2APart(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str | None = None
    data: dict[str, Any] | None = None
    media_type: str | None = Field(default=None, alias="mediaType")
    metadata: dict[str, Any] | None = None


class A2AMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message_id: str | None = Field(default=None, alias="messageId")
    role: str
    parts: list[A2APart]
    metadata: dict[str, Any] | None = None


class A2ATaskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId")
    context_id: str = Field(alias="contextId")
    messages: list[A2AMessage]
    metadata: dict[str, Any] | None = None


class A2ATaskStatus(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state: Literal[
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
        "TASK_STATE_FAILED",
        "TASK_STATE_REJECTED",
    ]
    timestamp: str
    message: A2AMessage | None = None


class A2AArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    artifact_id: str = Field(alias="artifactId")
    name: str | None = None
    description: str | None = None
    parts: list[A2APart]
    metadata: dict[str, Any] | None = None


class A2ATaskResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    context_id: str = Field(alias="contextId")
    status: A2ATaskStatus
    artifacts: list[A2AArtifact] = Field(default_factory=list)
    history: list[A2AMessage] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
