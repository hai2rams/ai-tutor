"""AI Tutor Platform — FastAPI entrypoint."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.tasks import TutorSessionError, run_tutor_session

load_dotenv()

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default")
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class TutorSessionRequest(BaseModel):
    student_id: str = Field(..., min_length=1, description="Student identifier within the tenant.")
    topic: str = Field(..., min_length=1, description="Curriculum topic or learning objective.")
    question: str = Field(..., min_length=1, description="Student question or prompt for the tutor crew.")


class TutorSessionResponse(BaseModel):
    tenant_id: str
    student_id: str
    topic: str
    answer: str
    status: str = "completed"


def resolve_tenant_id(
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
) -> str:
    """Resolve tenant from header; fall back to platform default."""
    tenant_id = (x_tenant_id or DEFAULT_TENANT_ID).strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="Tenant ID is required.")
    return tenant_id


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.getLogger(__name__).info("AI Tutor API starting (crew orchestration enabled).")
    yield


app = FastAPI(
    title="AI Tutor Platform",
    description="Multi-tenant hybrid AI tutor powered by CrewAI, ChromaDB, and Google Gemini.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="ai-tutor", version=app.version)


@app.post(f"{API_PREFIX}/sessions", response_model=TutorSessionResponse, tags=["tutor"])
async def create_tutor_session(
    payload: TutorSessionRequest,
    tenant_id: Annotated[str, Depends(resolve_tenant_id)],
) -> TutorSessionResponse:
    """Run the CrewAI tutoring workflow for a tenant-scoped student session."""
    try:
        answer = await asyncio.to_thread(
            run_tutor_session,
            tenant_id,
            payload.student_id,
            payload.topic,
            payload.question,
        )
    except TutorSessionError as exc:
        message = str(exc)
        status_code = 503 if "GEMINI_API_KEY" in message else 502
        raise HTTPException(status_code=status_code, detail=message) from exc

    return TutorSessionResponse(
        tenant_id=tenant_id,
        student_id=payload.student_id,
        topic=payload.topic,
        answer=answer,
        status="completed",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("APP_ENV", "development") == "development",
    )
