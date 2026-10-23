"""Local Ollama embedding provider."""

import math
from typing import Any

import httpx

from app.core.config import Settings
from app.providers.base import ProviderConnectionError, ProviderError, ProviderModelNotFoundError


class OllamaEmbeddingProvider:
    def __init__(self, settings: Settings, model: str = "nomic-embed-text:latest") -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(max(settings.request_timeout_seconds, 120.0))
        self._installed_verified = False

    async def ensure_installed(self) -> None:
        if self._installed_verified:
            return
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderConnectionError("Ollama embedding registry request timed out") from exc
        except (httpx.HTTPError, OSError) as exc:
            raise ProviderConnectionError("Could not connect to Ollama for embeddings") from exc
        except ValueError as exc:
            raise ProviderError("Ollama returned invalid model registry data") from exc
        models = payload.get("models", []) if isinstance(payload, dict) else []
        names = {item.get("name") for item in models if isinstance(item, dict)}
        identifiers = {item.get("model") for item in models if isinstance(item, dict)}
        if self.model not in names and self.model not in identifiers:
            raise ProviderModelNotFoundError(f"Embedding model '{self.model}' is not installed in Ollama")
        self._installed_verified = True

    async def embed(self, text: str) -> list[float]:
        await self.ensure_installed()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": text})
                if response.status_code == 404:
                    raise ProviderModelNotFoundError(f"Embedding model '{self.model}' is not installed in Ollama")
                response.raise_for_status()
                payload: Any = response.json()
        except (ProviderModelNotFoundError, ProviderError):
            raise
        except httpx.TimeoutException as exc:
            raise ProviderConnectionError("Ollama embedding request timed out") from exc
        except (httpx.HTTPError, OSError) as exc:
            raise ProviderConnectionError("Could not connect to Ollama for embeddings") from exc
        except ValueError as exc:
            raise ProviderError("Ollama returned invalid embedding JSON") from exc
        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        vector = embeddings[0] if isinstance(embeddings, list) and embeddings else None
        if not isinstance(vector, list) or not vector or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
            raise ProviderError("Ollama returned an invalid embedding vector")
        return [float(value) for value in vector]
