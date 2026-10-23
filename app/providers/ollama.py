"""Ollama provider using its local HTTP API."""

from datetime import datetime

import httpx

from app.core.config import Settings
from app.providers.base import (
    InferenceProvider,
    ProviderConnectionError,
    ProviderError,
    ProviderModelNotFoundError,
)
from app.schemas.chat import ChatResponse, OllamaTiming
from app.schemas.models import ModelInfo


class OllamaProvider(InferenceProvider):
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.timeout = httpx.Timeout(settings.request_timeout_seconds)

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.is_success
        except (httpx.HTTPError, OSError):
            return False

    async def list_models(self) -> list[ModelInfo]:
        payload = await self._get_json("/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise ProviderError("Ollama returned an invalid model registry response")
        return [self._model_info(item) for item in models]

    async def chat(self, model: str, message: str, *, options: dict | None = None, think: bool | None = None) -> ChatResponse:
        installed = await self.list_models()
        if not any(item.name == model or item.identifier == model for item in installed):
            raise ProviderModelNotFoundError(f"Model '{model}' is not installed in Ollama")
        payload = {"model": model, "messages": [{"role": "user", "content": message}], "stream": False}
        if options:
            payload["options"] = options
        if model.lower().split(":", 1)[0] == "qwen3" and think is not None:
            payload["think"] = think
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                if response.status_code == 404:
                    raise ProviderModelNotFoundError(f"Model '{model}' is not installed in Ollama")
                response.raise_for_status()
                data = response.json()
        except ProviderModelNotFoundError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderConnectionError("Ollama chat request timed out") from exc
        except (httpx.HTTPError, OSError) as exc:
            raise ProviderConnectionError("Could not connect to Ollama") from exc
        except ValueError as exc:
            raise ProviderError("Ollama returned invalid chat response data") from exc
        response_text = data.get("message", {}).get("content")
        if not isinstance(response_text, str):
            raise ProviderError("Ollama returned a response without message content")
        if not response_text.strip():
            # Qwen3 can return only a thinking trace when thinking is enabled.
            # Never expose that trace as the answer or let the UI show citations
            # without an explanation.
            raise ProviderError("Ollama returned an empty final answer")
        timing = OllamaTiming(
            total_ms=self._ns_ms(data.get("total_duration")),
            load_ms=self._ns_ms(data.get("load_duration")),
            prompt_eval_ms=self._ns_ms(data.get("prompt_eval_duration")),
            generation_eval_ms=self._ns_ms(data.get("eval_duration")),
            prompt_tokens=self._nonnegative_int(data.get("prompt_eval_count")),
            generated_tokens=self._nonnegative_int(data.get("eval_count")),
        )
        if timing.generated_tokens and timing.generation_eval_ms:
            timing.tokens_per_second = round(timing.generated_tokens / (timing.generation_eval_ms / 1000), 2)
        if all(value is None for value in timing.model_dump().values()):
            timing = None
        return ChatResponse(model=model, response=response_text, done=bool(data.get("done", True)), timing=timing)

    @staticmethod
    def _ns_ms(value: object) -> int | None:
        return round(value / 1_000_000) if isinstance(value, (int, float)) and value >= 0 else None

    @staticmethod
    def _nonnegative_int(value: object) -> int | None:
        return int(value) if isinstance(value, int) and value >= 0 else None

    async def _get_json(self, path: str) -> dict[str, object]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}{path}")
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderConnectionError("Ollama request timed out") from exc
        except (httpx.HTTPError, OSError) as exc:
            raise ProviderConnectionError("Could not connect to Ollama") from exc
        except ValueError as exc:
            raise ProviderError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ProviderError("Ollama returned an invalid response")
        return data

    @staticmethod
    def _model_info(item: object) -> ModelInfo:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ProviderError("Ollama returned an invalid model entry")
        modified_at = item.get("modified_at")
        parsed_modified_at = None
        if isinstance(modified_at, str):
            try:
                parsed_modified_at = datetime.fromisoformat(modified_at.replace("Z", "+00:00"))
            except ValueError:
                parsed_modified_at = None
        details = item.get("details") if isinstance(item.get("details"), dict) else None
        capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), list) else []
        return ModelInfo(
            name=item["name"],
            identifier=str(item.get("model", item["name"])),
            digest=item.get("digest") if isinstance(item.get("digest"), str) else None,
            size=item.get("size") if isinstance(item.get("size"), int) else None,
            modified_at=parsed_modified_at,
            details=details,
            capabilities=[value for value in capabilities if isinstance(value, str)],
        )
