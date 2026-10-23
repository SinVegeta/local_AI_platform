from app.registry.model_registry import ModelRegistry
from app.schemas.models import ModelInfo


def runtime_models():
    return [
        ModelInfo(name="qwen2.5:3b", identifier="qwen2.5:3b", capabilities=["completion", "tools"], size=100),
        ModelInfo(name="nomic-embed-text:latest", identifier="nomic-embed-text:latest", capabilities=["embedding"], size=50),
        ModelInfo(name="llama3.2:3b", identifier="llama3.2:3b", capabilities=["completion"], size=120),
    ]


def test_manifest_has_exact_catalog_identifiers_and_verified_license_metadata():
    registry = ModelRegistry()
    catalog = {model.identifier: model for model in registry.list_models([])}
    assert {"qwen3:4b", "granite4.1:3b", "qwen2.5-coder:1.5b", "glm-ocr", "nomic-embed-text", "qwen2.5:3b"} <= set(catalog)
    assert catalog["qwen3:4b"].license_info.status == "verified_permissive"
    assert catalog["qwen2.5:3b"].license_info.status == "verified_research_restricted"


def test_installed_status_uses_ollama_as_authority_and_latest_alias():
    registry = ModelRegistry()
    models = registry.list_models(runtime_models())
    by_id = {model.identifier: model for model in models}
    assert by_id["qwen2.5:3b"].installed is True
    assert by_id["nomic-embed-text:latest"].installed is True
    assert by_id["qwen3:4b"].installed is False
    assert by_id["llama3.2:3b"].compatibility_status == "unverified_runtime"


def test_task_filter_and_preferred_selection():
    registry = ModelRegistry()
    coding = registry.list_models(runtime_models(), task="coding")
    assert all("coding" in model.supported_tasks for model in coding)
    assert any(model.identifier == "qwen2.5:3b" and model.installed for model in coding)
    assert registry.task("chat").preferred_model == "qwen3:4b"


def test_pipeline_metadata_matches_implemented_local_workflows():
    registry = ModelRegistry()
    tasks = {task.id: task for task in registry.list_tasks()}
    assert tasks["ocr"].pipeline_available is True
    assert tasks["embeddings"].pipeline_available is True

