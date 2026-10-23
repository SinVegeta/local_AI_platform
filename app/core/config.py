"""Environment-backed application settings."""

import os
from functools import lru_cache

from pydantic import BaseModel, Field


class Settings(BaseModel):
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    api_prefix: str = Field(default="/api/v1")
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    frontend_origins: list[str] = Field(
        default=["http://127.0.0.1:5173", "http://localhost:5173"]
    )
    max_render_dpi: int = Field(default=300, ge=72, le=300)
    tesseract_path: str = Field(default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    tesseract_timeout_seconds: float = Field(default=120.0, gt=0)
    rag_min_similarity: float = Field(default=0.25, ge=0.0, le=1.0)
    rag_storage_path: str = Field(default="data/rag/index.sqlite3")
    rag_num_predict: int = Field(default=1024, ge=32, le=2048)
    rag_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # Direct answers are the default for document Q&A. Users can opt into
    # Qwen3 thinking with RAG_QWEN3_THINK=true or the request-level flag.
    rag_qwen3_think: bool | None = False

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.model_fields["ollama_base_url"].default),
            api_prefix=os.getenv("API_PREFIX", cls.model_fields["api_prefix"].default),
            request_timeout_seconds=float(
                os.getenv(
                    "REQUEST_TIMEOUT_SECONDS",
                    str(cls.model_fields["request_timeout_seconds"].default),
                )
            ),
            frontend_origins=[
                origin.strip()
                for origin in os.getenv(
                    "FRONTEND_ORIGINS",
                    "http://127.0.0.1:5173,http://localhost:5173",
                ).split(",")
                if origin.strip()
            ],
            max_render_dpi=int(os.getenv("MAX_RENDER_DPI", "300")),
            tesseract_path=os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            tesseract_timeout_seconds=float(os.getenv("TESSERACT_TIMEOUT_SECONDS", "120")),
            rag_min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.25")),
            rag_storage_path=os.getenv("RAG_STORAGE_PATH", "data/rag/index.sqlite3"),
            rag_num_predict=int(os.getenv("RAG_NUM_PREDICT", "1024")),
            rag_temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
            rag_qwen3_think=os.getenv("RAG_QWEN3_THINK", "false").lower() == "true",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_environment()
