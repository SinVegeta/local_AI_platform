"""Conservative local document-processing settings."""

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


DOCUMENT_ROOT = Path(os.getenv("LOCAL_DOCUMENT_ROOT", "data/documents")).resolve()
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 100)
MAX_RENDER_DPI = _int("MAX_RENDER_DPI", 300)
MAX_OCR_SECONDS = _int("MAX_OCR_SECONDS", 900)
MAX_CONCURRENT_OCR_JOBS = 1
OCR_WORKER_PYTHON = os.getenv("OCR_WORKER_PYTHON", ".ocr-venv/Scripts/python.exe")
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSERACT_TIMEOUT_SECONDS = _int("TESSERACT_TIMEOUT_SECONDS", 120)
