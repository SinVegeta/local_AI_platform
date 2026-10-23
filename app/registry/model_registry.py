"""Validated catalog joined with Ollama's authoritative installed registry."""

import json
from pathlib import Path
from typing import Any

from app.registry.lifecycle import ModelLifecyclePolicy
from app.schemas.models import LicenseInfo, ModelInfo, TaskInfo

MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")


def _latest_normalized(identifier: str) -> str:
    return identifier.removesuffix(":latest")


class ModelRegistry:
    def __init__(self, manifest_path: Path = MANIFEST_PATH) -> None:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            self.manifest: dict[str, Any] = json.load(manifest_file)
        self.lifecycle = ModelLifecyclePolicy(
            max_concurrent_generation_models=self.manifest["runtime_policy"]["max_concurrent_generation_models"],
            load_behavior=self.manifest["runtime_policy"]["load_behavior"],
            automatic_downloads=self.manifest["runtime_policy"]["automatic_downloads"],
        )
        self._tasks = {task["id"]: task for task in self.manifest["tasks"]}
        self._catalog = {model["ollama_identifier"]: model for model in self.manifest["models"]}

    def list_tasks(self) -> list[TaskInfo]:
        return [TaskInfo(**task) for task in self.manifest["tasks"]]

    def task(self, task_id: str) -> TaskInfo:
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task '{task_id}'")
        return TaskInfo(**task)

    def list_models(self, installed: list[ModelInfo], task: str | None = None) -> list[ModelInfo]:
        installed_by_normalized = {_latest_normalized(model.identifier): model for model in installed}
        result: list[ModelInfo] = []
        catalog_identifiers = set()
        for identifier, catalog in self._catalog.items():
            catalog_identifiers.add(_latest_normalized(identifier))
            runtime = installed_by_normalized.get(_latest_normalized(identifier))
            supported_tasks = list(catalog["supported_tasks"])
            if task and task not in supported_tasks:
                continue
            result.append(self._join_catalog(catalog, runtime, supported_tasks))

        for runtime in installed:
            normalized = _latest_normalized(runtime.identifier)
            if normalized in catalog_identifiers:
                continue
            inferred_tasks = self._infer_tasks(runtime)
            if task and task not in inferred_tasks:
                continue
            result.append(
                runtime.model_copy(
                    update={
                        "display_name": runtime.name,
                        "provider": "ollama",
                        "supported_tasks": inferred_tasks,
                        "supported_input_modalities": ["text"],
                        "license_info": LicenseInfo(
                            identifier=None,
                            status="unverified",
                            official_model_card_url=None,
                            source_url=None,
                        ),
                        "installed": True,
                        "runtime_status": "unregistered_installed",
                        "compatibility_status": "unverified_runtime",
                        "availability_warning": "Installed in Ollama but not present in the curated catalog; task compatibility is inferred from runtime metadata.",
                        "verification_date": self.manifest["verification_date"],
                    }
                )
            )
        return result

    def find_model(self, identifier: str, installed: list[ModelInfo], task: str) -> ModelInfo | None:
        return next(
            (
                model
                for model in self.list_models(installed, task=task)
                if model.installed
                and (
                    model.identifier == identifier
                    or model.name == identifier
                    or _latest_normalized(model.identifier) == _latest_normalized(identifier)
                )
            ),
            None,
        )

    def _join_catalog(self, catalog: dict[str, Any], runtime: ModelInfo | None, supported_tasks: list[str]) -> ModelInfo:
        installed = runtime is not None
        return ModelInfo(
            name=runtime.name if runtime else catalog["ollama_identifier"],
            identifier=runtime.identifier if runtime else catalog["ollama_identifier"],
            digest=runtime.digest if runtime else None,
            size=runtime.size if runtime else None,
            modified_at=runtime.modified_at if runtime else None,
            details=runtime.details if runtime else None,
            capabilities=runtime.capabilities if runtime else [],
            display_name=catalog["display_name"],
            provider=catalog["provider"],
            supported_tasks=supported_tasks,
            supported_input_modalities=catalog["supported_input_modalities"],
            approximate_model_size=catalog["approximate_model_size"],
            required_vram_estimate=catalog["required_vram_estimate"],
            license_info=LicenseInfo(
                identifier=catalog["license_identifier"],
                status=catalog["license_status"],
                official_model_card_url=catalog["official_model_card_url"],
                source_url=catalog["license_source_url"],
            ),
            installed=installed,
            runtime_status=self.lifecycle.status_for(installed=installed, task=supported_tasks[0]),
            compatibility_status=catalog["compatibility_status"],
            availability_warning=None if installed else f"Preferred catalog model is not installed: {catalog['ollama_identifier']}",
            verification_date=self.manifest["verification_date"],
            notes=catalog["notes"],
        )

    @staticmethod
    def _infer_tasks(model: ModelInfo) -> list[str]:
        if "embedding" in model.capabilities or "embed" in model.name.lower():
            return ["embeddings"]
        if "completion" in model.capabilities:
            return ["chat"]
        return []
