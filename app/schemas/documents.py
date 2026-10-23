"""Document ingestion and OCR response schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DocumentStatus = Literal["uploaded", "queued", "processing", "completed", "partially_completed", "failed", "interrupted"]
PageStatus = Literal["pending", "processing", "completed", "failed"]


class OCRConfig(BaseModel):
    language: str = Field(default="en", min_length=2, max_length=16)
    primary_engine: Literal["paddleocr", "tesseract", "glm-ocr"] = "paddleocr"
    enable_fallback: bool = True
    fallback_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    render_dpi: int = Field(default=150, ge=72, le=300)
    force_ocr: bool = False


class TextBlock(BaseModel):
    text: str
    bbox: list[float] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class PageResult(BaseModel):
    page_number: int
    status: PageStatus
    text: str = ""
    blocks: list[TextBlock] = []
    extraction_method: str | None = None
    engine: str | None = None
    engine_version: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = []
    error: str | None = None
    source_image: str | None = None
    duration_ms: int | None = None
    fallback_text: str | None = None
    fallback_engine: str | None = None
    primary_text: str | None = None
    primary_engine: str | None = None
    primary_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    selection_reason: str | None = None


class DocumentSummary(BaseModel):
    document_id: str
    original_filename: str
    media_type: str
    size_bytes: int
    page_count: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    completed_pages: int = 0
    warnings: list[str] = []


class DocumentDetail(DocumentSummary):
    config: OCRConfig
    pages: list[PageResult] = []
    processing_started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class DocumentTextResponse(BaseModel):
    document_id: str
    text: str
    pages: list[PageResult]
