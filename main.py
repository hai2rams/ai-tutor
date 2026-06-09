"""AI Tutor Platform — A2A control plane with Gemini + CrewAI execution."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.a2a.agent_card import build_agent_card
from src.a2a.schemas import A2ATaskRequest, A2ATaskResponse
from src.a2a.task_handler import (
    A2ATaskParseError,
    build_completed_task_response,
    parse_tutoring_session,
)
from src.tasks import TutorSessionError, run_tutor_session
from utils.db_sync import derive_session_score, update_student_mastery

load_dotenv()

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
A2A_AGENT_BASE_PATH = "/a2a/tutor-agent"
logging.basicConfig(level=LOG_LEVEL)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    protocols: list[str] = Field(default_factory=lambda: ["A2A", "MCP"])


def resolve_tenant_id(
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
) -> str:
    """Resolve tenant from header; fall back to platform default."""
    tenant_id = (x_tenant_id or DEFAULT_TENANT_ID).strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID is required.")
    return tenant_id


def _public_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.getLogger(__name__).info(
        "AI Tutor A2A control plane starting (Gemini crew + MCP tool discovery)."
    )
    yield


app = FastAPI(
    title="AI Tutor Platform",
    description=(
        "Multi-tenant hybrid AI tutor exposing an A2A control plane, "
        "CrewAI execution on Google Gemini, and MCP-backed knowledge tools."
    ),
    version=os.getenv("APP_VERSION", "0.2.0"),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="ai-tutor",
        version=app.version,
        protocols=["A2A", "MCP"],
    )


@app.get(f"{A2A_AGENT_BASE_PATH}/agent.json", tags=["a2a"])
async def get_agent_card(request: Request) -> dict[str, Any]:
    """Expose the official A2A Agent Card for discovery and orchestration."""
    return build_agent_card(base_url=f"{_public_base_url(request)}{A2A_AGENT_BASE_PATH}")


@app.post(
    f"{A2A_AGENT_BASE_PATH}/tasks",
    response_model=A2ATaskResponse,
    response_model_by_alias=True,
    tags=["a2a"],
)
async def execute_a2a_task(
    payload: A2ATaskRequest,
    tenant_id: Annotated[str, Depends(resolve_tenant_id)],
) -> A2ATaskResponse:
    """Accept a standardized A2A task payload and run the tutoring crew."""
    try:
        student_id, topic, question = parse_tutoring_session(payload)
    except A2ATaskParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        answer = await asyncio.to_thread(
            run_tutor_session,
            tenant_id,
            student_id,
            topic,
            question,
        )
    except TutorSessionError as exc:
        message = str(exc)
        status_code = 503 if "GEMINI_API_KEY" in message else 502
        raise HTTPException(status_code=status_code, detail=message) from exc

    session_score = derive_session_score(question, answer)
    mastery_record = await asyncio.to_thread(
        update_student_mastery,
        tenant_id,
        student_id,
        topic,
        session_score,
        answer,
    )

    return build_completed_task_response(
        payload,
        answer,
        tenant_id=tenant_id,
        session_score=session_score,
        mastery_status=mastery_record.get("status", "UNKNOWN"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("APP_ENV", "development") == "development",
    )
