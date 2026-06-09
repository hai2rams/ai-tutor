"""A2A Agent Card builder for the tutor agent."""

from __future__ import annotations

import os
from typing import Any


def build_agent_card(base_url: str | None = None) -> dict[str, Any]:
    """Return the structured Agent Card served at /a2a/tutor-agent/agent.json."""
    host = os.getenv("A2A_PUBLIC_HOST", os.getenv("HOST", "localhost"))
    port = os.getenv("A2A_PUBLIC_PORT", os.getenv("PORT", "8000"))
    interface_base = base_url or f"http://{host}:{port}/a2a/tutor-agent"

    return {
        "protocolVersion": "1.0",
        "name": "AI Tutor Agent",
        "description": (
            "Multi-tenant hybrid tutoring agent powered by CrewAI and Google Gemini. "
            "Discovers curriculum tools via MCP and returns MOE-aligned tutoring responses."
        ),
        "version": os.getenv("APP_VERSION", "0.2.0"),
        "provider": {
            "organization": "AI Tutor Platform",
            "url": interface_base,
        },
        "supportedInterfaces": [
            {
                "url": interface_base,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "moe-tutoring",
                "name": "MOE Curriculum Tutoring",
                "description": (
                    "Provides Singapore MOE-aligned tutoring with curriculum lookup, "
                    "reading support, misconception review, and mastery tracking."
                ),
                "tags": ["education", "tutoring", "moe", "gemini", "mcp"],
                "examples": [
                    "Explain equivalent fractions to a Primary 4 student.",
                    "Help a student understand 1/2 + 1/4 with a worked example.",
                ],
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["text/plain", "application/json"],
            }
        ],
        "securitySchemes": {
            "tenantHeader": {
                "apiKeySecurityScheme": {
                    "name": "X-Tenant-ID",
                    "location": "header",
                    "description": "Tenant routing header for multi-tenant isolation.",
                }
            }
        },
        "securityRequirements": [{"tenantHeader": []}],
    }
