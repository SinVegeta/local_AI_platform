import httpx
import pytest

from app.core.config import Settings
from app.providers.base import ProviderError, ProviderModelNotFoundError
from app.providers.embeddings import OllamaEmbeddingProvider


@pytest.mark.asyncio
async def test_embedding_provider_reads_dimension_and_rejects_bad_vectors():
    responses = {
        "/api/tags": httpx.Response(200, json={"models": [{"name": "nomic-embed-text:latest"}]}),
        "/api/embed": httpx.Response(200, json={"embeddings": [[1, 2, 3]]}),
    }
    transport = httpx.MockTransport(lambda request: responses[request.url.path])
    original = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        provider = OllamaEmbeddingProvider(Settings())
        vector = await provider.embed("hello")
        assert vector == [1.0, 2.0, 3.0]
    finally:
        httpx.AsyncClient = original


@pytest.mark.asyncio
async def test_embedding_provider_reports_missing_model():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": []}))
    original = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        with pytest.raises(ProviderModelNotFoundError):
            await OllamaEmbeddingProvider(Settings()).embed("hello")
    finally:
        httpx.AsyncClient = original
