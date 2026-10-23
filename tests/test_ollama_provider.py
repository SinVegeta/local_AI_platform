import httpx
import pytest

from app.core.config import Settings
from app.providers.ollama import OllamaProvider


@pytest.mark.asyncio
async def test_provider_lists_models_with_mock_transport():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": [{"name": "demo:1b", "model": "demo:1b", "digest": "abc", "size": 123}]}))
    provider = OllamaProvider(Settings())
    original = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        models = await provider.list_models()
        assert models[0].identifier == "demo:1b"
    finally:
        httpx.AsyncClient = original


@pytest.mark.asyncio
async def test_provider_parses_ollama_timing_and_qwen3_think_option():
    captured = {}

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b", "model": "qwen3:4b"}]})
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"message": {"content": "grounded"}, "done": True, "total_duration": 2_000_000_000, "load_duration": 500_000_000, "prompt_eval_duration": 300_000_000, "eval_duration": 1_200_000_000, "prompt_eval_count": 40, "eval_count": 12})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        result = await OllamaProvider(Settings()).chat("qwen3:4b", "context", options={"num_predict": 64}, think=False)
        assert captured["think"] is False
        assert captured["options"]["num_predict"] == 64
        assert result.timing.total_ms == 2000
        assert result.timing.load_ms == 500
        assert result.timing.prompt_eval_ms == 300
        assert result.timing.generation_eval_ms == 1200
        assert result.timing.tokens_per_second == 10.0
    finally:
        httpx.AsyncClient = original


@pytest.mark.asyncio
async def test_provider_omits_think_for_non_qwen3_models():
    captured = {}

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "granite4.1:3b", "model": "granite4.1:3b"}]})
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        result = await OllamaProvider(Settings()).chat("granite4.1:3b", "context", think=False)
        assert "think" not in captured
        assert result.timing is None
    finally:
        httpx.AsyncClient = original
