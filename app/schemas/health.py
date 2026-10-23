"""Health response model."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
