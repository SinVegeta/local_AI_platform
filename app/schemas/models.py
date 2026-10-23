"""Installed model response models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

TaskName = Literal["chat", "coding", "ocr", "embeddings"]


class LicenseInfo(BaseModel):
    identifier: str | None = None
    status: str
    official_model_card_url: str | None = None
    source_url: str | None = None


class ModelInfo(BaseModel):
    name: str
    identifier: str
    digest: str | None = None
    size: int | None = None
    modified_at: datetime | None = None
    details: dict[str, object] | None = None
    capabilities: list[str] = []
    display_name: str | None = None
    provider: str = "ollama"
    supported_tasks: list[str] = []
    supported_input_modalities: list[str] = ["text"]
    approximate_model_size: str | None = None
    required_vram_estimate: str | None = None
    license_info: LicenseInfo | None = None
    installed: bool = True
    runtime_status: str = "installed_not_loaded"
    compatibility_status: str = "unverified_runtime"
    availability_warning: str | None = None
    verification_date: str | None = None
    notes: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class TaskInfo(BaseModel):
    id: TaskName
    display_name: str
    description: str
    preferred_model: str
    pipeline_available: bool
    pipeline_note: str
