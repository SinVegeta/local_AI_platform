"""Use-case services independent of any particular runtime."""

from app.providers.base import InferenceProvider
from app.providers.base import (
    InvalidTaskError,
    ProviderModelIncompatibleError,
    ProviderModelNotFoundError,
    TaskPipelineUnavailableError,
)
from app.registry.model_registry import ModelRegistry
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.models import ModelInfo, TaskInfo


class PlatformService:
    def __init__(self, provider: InferenceProvider, registry: ModelRegistry | None = None) -> None:
        self.provider = provider
        self.registry = registry or ModelRegistry()

    async def check_provider(self) -> bool:
        return await self.provider.is_available()

    async def list_models(self, task: str | None = None) -> list[ModelInfo]:
        installed = await self.provider.list_models()
        return self.registry.list_models(installed, task=task)

    def list_tasks(self) -> list[TaskInfo]:
        return self.registry.list_tasks()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        try:
            task = self.registry.task(request.task)
        except ValueError as exc:
            raise InvalidTaskError(str(exc)) from exc
        if not task.pipeline_available:
            raise TaskPipelineUnavailableError(
                f"The {task.display_name} pipeline is not implemented yet. {task.pipeline_note}"
            )
        installed = await self.provider.list_models()
        selected = self.registry.find_model(request.model, installed, request.task)
        if selected is None:
            installed_match = next(
                (
                    model
                    for model in self.registry.list_models(installed)
                    if model.installed and (model.identifier == request.model or model.name == request.model)
                ),
                None,
            )
            if installed_match is not None:
                raise ProviderModelIncompatibleError(
                    f"Model '{request.model}' is not compatible with task '{request.task}'. "
                    f"Compatible installed models: {', '.join(self._compatible_names(installed, request.task)) or 'none'}"
                )
            raise ProviderModelNotFoundError(
                f"Model '{request.model}' is not installed in Ollama for task '{request.task}'. "
                f"Preferred model: '{task.preferred_model}'."
            )
        return await self.provider.chat(request.model, request.message)

    def _compatible_names(self, installed: list[ModelInfo], task: str) -> list[str]:
        return [model.identifier for model in self.registry.list_models(installed, task=task) if model.installed]
