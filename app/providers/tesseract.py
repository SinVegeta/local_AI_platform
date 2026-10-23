"""Optional local CPU OCR provider backed by the installed Tesseract binary."""

import re
import subprocess
from pathlib import Path

from app.core.config import Settings
from app.providers.ocr_base import OCRProvider, OCRProviderError
from app.schemas.documents import OCRConfig, PageResult


class TesseractProvider(OCRProvider):
    name = "tesseract"

    def __init__(self, settings: Settings, executable: str | None = None, timeout: float | None = None) -> None:
        self.executable = executable or settings.tesseract_path
        self.timeout = timeout or settings.tesseract_timeout_seconds
        self._version: str | None = None

    def extract(self, image_path: Path, page_number: int, config: OCRConfig) -> PageResult:
        language = self._tesseract_language(config.language)
        self._ensure_language(language)
        version = self.version()
        command = [self.executable, str(image_path), "stdout", "--dpi", "300", "-l", language, "--psm", "6"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout, check=False)
        except FileNotFoundError as exc:
            raise OCRProviderError("Tesseract executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise OCRProviderError("Tesseract OCR timed out") from exc
        except OSError as exc:
            raise OCRProviderError("Tesseract could not be started") from exc
        if completed.returncode != 0:
            raise OCRProviderError("Tesseract could not process the image")
        text = completed.stdout.strip()
        warnings = ["Tesseract OCR runs on CPU; no GPU acceleration is used."]
        if not text:
            warnings.append("Tesseract completed but recognized no text.")
        return PageResult(page_number=page_number, status="completed", text=text, extraction_method="ocr", engine=self.name,
                          engine_version=version, warnings=warnings)

    def installed_languages(self) -> set[str]:
        try:
            completed = subprocess.run([self.executable, "--list-langs"], capture_output=True, text=True, timeout=self.timeout, check=False)
        except FileNotFoundError as exc:
            raise OCRProviderError("Tesseract executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise OCRProviderError("Tesseract language discovery timed out") from exc
        except OSError as exc:
            raise OCRProviderError("Tesseract could not be started") from exc
        if completed.returncode != 0:
            raise OCRProviderError("Tesseract language discovery failed")
        return {line.strip() for line in completed.stdout.splitlines() if re.fullmatch(r"[A-Za-z0-9_]+", line.strip())}

    def _ensure_language(self, language: str) -> None:
        if language not in self.installed_languages():
            raise OCRProviderError(f"Tesseract language '{language}' is not installed")

    @staticmethod
    def _tesseract_language(language: str) -> str:
        # The UI/API uses the common ISO-style `en`; Tesseract's installed
        # English traineddata is conventionally named `eng`.
        return "eng" if language == "en" else language

    def version(self) -> str:
        if self._version:
            return self._version
        try:
            completed = subprocess.run([self.executable, "--version"], capture_output=True, text=True, timeout=self.timeout, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            raise OCRProviderError("Tesseract version could not be read") from exc
        if completed.returncode != 0:
            raise OCRProviderError("Tesseract version could not be read")
        self._version = completed.stdout.splitlines()[0].strip() if completed.stdout else "unknown"
        return self._version
