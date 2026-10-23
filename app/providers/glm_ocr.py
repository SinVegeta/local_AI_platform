"""Local Ollama GLM-OCR image fallback."""

import base64
from pathlib import Path

import httpx

from app.core.config import Settings
from app.providers.ocr_base import OCRProvider, OCRProviderError
from app.schemas.documents import OCRConfig, PageResult


class GLMOCRProvider(OCRProvider):
    name = "glm-ocr"

    def __init__(self, settings: Settings, model: str = "glm-ocr:latest") -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(max(settings.request_timeout_seconds, 180.0))

    def extract(self, image_path: Path, page_number: int, config: OCRConfig) -> PageResult:
        try:
            image = base64.b64encode(image_path.read_bytes()).decode("ascii")
            payload = {"model": self.model, "messages": [{"role": "user", "content": "Transcribe all visible text exactly. Return only the transcription.", "images": [image]}], "stream": False}
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                if response.status_code == 404:
                    raise OCRProviderError("GLM-OCR model is not installed in Ollama")
                response.raise_for_status()
                data = response.json()
            text = data.get("message", {}).get("content")
            if not isinstance(text, str):
                raise OCRProviderError("GLM-OCR returned no transcription")
            return PageResult(page_number=page_number, status="completed", text=text.strip(), extraction_method="vision_fallback", engine=self.name,
                              engine_version=self.model, warnings=["Fallback output is retained separately from the primary OCR result."])
        except OCRProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise OCRProviderError("GLM-OCR request timed out") from exc
        except (httpx.HTTPError, OSError, ValueError) as exc:
            raise OCRProviderError("GLM-OCR local request failed") from exc
