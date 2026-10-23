"""Provider interface and provider-level exceptions."""

from abc import ABC, abstractmethod

from app.schemas.chat import ChatResponse
from app.schemas.models import ModelInfo


class ProviderError(RuntimeError):
    """Base error raised by inference providers."""


class ProviderConnectionError(ProviderError):
    """Provider could not be reached."""


class ProviderModelNotFoundError(ProviderError):
    """Requested model is not installed."""


class ProviderModelIncompatibleError(ProviderError):
    """Requested model cannot serve the selected task."""


class TaskPipelineUnavailableError(ProviderError):
    """The catalog task exists but its execution pipeline is not implemented."""


class InvalidTaskError(ProviderError):
    """Task is not present in the model catalog."""


class InferenceProvider(ABC):
    @abstractmethod
    async def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    @abstractmethod
    async def chat(self, model: str, message: str) -> ChatResponse:
        raise NotImplementedError
