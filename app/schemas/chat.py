"""Chat request and response models."""

from pydantic import BaseModel, Field

from app.schemas.models import TaskName


class ChatRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=100_000)
    task: TaskName = "chat"


class OllamaTiming(BaseModel):
    total_ms: int | None = None
    load_ms: int | None = None
    prompt_eval_ms: int | None = None
    generation_eval_ms: int | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    tokens_per_second: float | None = None


class ChatResponse(BaseModel):
    model: str
    response: str
    done: bool = True
    timing: OllamaTiming | None = None
