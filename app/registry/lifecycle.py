"""Small, explicit model lifecycle policy for constrained local hardware."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelLifecyclePolicy:
    max_concurrent_generation_models: int = 1
    load_behavior: str = "ollama_managed"
    automatic_downloads: bool = False

    def status_for(self, *, installed: bool, task: str) -> str:
        if not installed:
            return "not_installed"
        if task in {"chat", "coding"}:
            return "installed_not_loaded"
        return "installed_runtime_managed"

    @property
    def description(self) -> str:
        return (
            "One generation model is preferred at a time. Ollama manages loading and may "
            "use CPU/GPU offloading; this service does not claim a model is loaded or unload "
            "unrelated processes."
        )
