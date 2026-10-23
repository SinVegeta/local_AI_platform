"""Provider-neutral OCR contracts."""

from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.documents import OCRConfig, PageResult


class OCRProviderError(RuntimeError):
    pass


class OCRProvider(ABC):
    name = "unknown"

    @abstractmethod
    def extract(self, image_path: Path, page_number: int, config: OCRConfig) -> PageResult:
        raise NotImplementedError
