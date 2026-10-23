"""PaddleOCR adapter; execution is isolated in .ocr-venv."""

import json
import subprocess
from pathlib import Path

from app.core.document_settings import OCR_WORKER_PYTHON
from app.providers.ocr_base import OCRProvider, OCRProviderError
from app.schemas.documents import OCRConfig, PageResult, TextBlock


class PaddleOCRProvider(OCRProvider):
    name = "paddleocr"

    def extract(self, image_path: Path, page_number: int, config: OCRConfig) -> PageResult:
        try:
            completed = subprocess.run(
                [OCR_WORKER_PYTHON, "-m", "app.providers.paddleocr_worker"],
                input=json.dumps({"image_path": str(image_path), "language": config.language}),
                capture_output=True, text=True, timeout=300, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OCRProviderError("PaddleOCR worker is unavailable or timed out") from exc
        try:
            payload = json.loads(completed.stdout)
        except ValueError as exc:
            raise OCRProviderError("PaddleOCR worker returned invalid output") from exc
        if not payload.get("ok"):
            raise OCRProviderError(payload.get("error", "PaddleOCR failed"))
        blocks = [TextBlock(**block) for block in payload.get("blocks", [])]
        confidence = sum(block.confidence for block in blocks if block.confidence is not None) / len(blocks) if blocks else None
        return PageResult(page_number=page_number, status="completed", text=payload.get("text", ""), blocks=blocks,
                          extraction_method="ocr", engine=self.name, engine_version=payload.get("engine_version"), confidence=confidence,
                          warnings=[f"PaddleOCR device: {payload.get('device', 'unknown')}"])
