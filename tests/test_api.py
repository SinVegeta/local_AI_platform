from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import get_service
from app.schemas.chat import ChatResponse
from app.schemas.models import ModelInfo


class FakeProvider:
    async def is_available(self):
        return True

    async def list_models(self):
        return [ModelInfo(name="qwen2.5:3b", identifier="qwen2.5:3b", digest="abc")]

    async def chat(self, model, message):
        return ChatResponse(model=model, response=f"Echo: {message}")


def test_health_and_models():
    app.dependency_overrides[get_service] = lambda: __import__("app.services.platform", fromlist=["PlatformService"]).PlatformService(FakeProvider())
    try:
        client = TestClient(app)
        assert client.get("/api/v1/health").json() == {"status": "ok", "ollama_connected": True}
        models = client.get("/api/v1/models").json()["models"]
        installed = {model["identifier"] for model in models if model["installed"]}
        assert "qwen2.5:3b" in installed
        assert client.get("/api/v1/tasks").json()[0]["id"] == "chat"
    finally:
        app.dependency_overrides.clear()


def test_chat_and_validation():
    app.dependency_overrides[get_service] = lambda: __import__("app.services.platform", fromlist=["PlatformService"]).PlatformService(FakeProvider())
    try:
        client = TestClient(app)
        response = client.post("/api/v1/chat", json={"model": "qwen2.5:3b", "message": "hello"})
        assert response.status_code == 200
        assert response.json()["response"] == "Echo: hello"
        assert client.post("/api/v1/chat", json={"model": "", "message": "hello"}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_unavailable_model_and_non_chat_model_are_explicit():
    app.dependency_overrides[get_service] = lambda: __import__("app.services.platform", fromlist=["PlatformService"]).PlatformService(FakeProvider())
    try:
        client = TestClient(app)
        missing = client.post("/api/v1/chat", json={"model": "qwen3:4b", "message": "hello", "task": "chat"})
        assert missing.status_code == 404
        assert "not installed" in missing.json()["detail"]
        unavailable = client.post("/api/v1/chat", json={"model": "nomic-embed-text", "message": "hello", "task": "embeddings"})
        assert unavailable.status_code == 404
        assert "not installed" in unavailable.json()["detail"]
        invalid_task = client.post("/api/v1/chat", json={"model": "qwen2.5:3b", "message": "hello", "task": "unknown"})
        assert invalid_task.status_code == 422
    finally:
        app.dependency_overrides.clear()
