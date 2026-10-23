from pathlib import Path

import pytest

from app.schemas.documents import OCRConfig, PageResult
from app.schemas.rag import RagQueryRequest
from app.schemas.rag import RagIndexResponse, RagQueryResponse
from app.api.routes import get_rag
from app.main import app
from fastapi.testclient import TestClient
from app.services.document_store import DocumentStore
from app.services.rag_chunking import chunk_pages
from app.services.rag_service import RagService
from app.services.rag_store import RagStore
from app.services.documents import DocumentService
from app.core.config import Settings


def test_page_aware_chunking_preserves_source_metadata():
    chunks = chunk_pages("doc", [{"page_number": 4, "text": "Alpha paragraph. Beta paragraph.", "extraction_method": "ocr", "engine": "tesseract"}], chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 4
    assert chunks[0]["extraction_method"] == "ocr"
    assert chunks[0]["content_hash"]


class FakeEmbeddings:
    model = "nomic-embed-text:latest"

    async def ensure_installed(self):
        return None

    async def embed(self, text):
        return [float(len(text)), 1.0, 0.0]


class FakeChat:
    async def chat(self, model, message, **kwargs):
        from app.schemas.chat import ChatResponse
        assert "Evidence" in message
        return ChatResponse(model=model, response="The evidence says Alpha.")


@pytest.mark.asyncio
async def test_index_retrieve_citation_and_delete(tmp_path: Path):
    documents = DocumentStore(tmp_path / "documents")
    source = tmp_path / "source.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\nlocal")
    detail = documents.create("source.png", "image/png", source.stat().st_size, 1, source, OCRConfig())
    detail.status = "completed"
    detail.pages[0] = PageResult(page_number=1, status="completed", text="Alpha is on page one.", extraction_method="ocr", engine="tesseract")
    documents.save(detail)
    service = RagService(None, document_store=documents, vector_store=RagStore(tmp_path / "index.sqlite3"), embeddings=FakeEmbeddings(), chat=FakeChat())
    indexed = await service.index([detail.document_id])
    assert indexed.indexes[0]["status"] == "indexed"
    result = await service.query(RagQueryRequest(question="Where is Alpha?", model="qwen2.5:3b", document_ids=[detail.document_id]))
    assert result.citations[0].document_id == detail.document_id
    assert result.citations[0].page_number == 1
    assert result.citations[0].excerpt == "Alpha is on page one."
    assert service.delete_index(detail.document_id) is True
    empty = await service.query(RagQueryRequest(question="Where is Alpha?", model="qwen2.5:3b", document_ids=[detail.document_id]))
    assert empty.citations == []


@pytest.mark.asyncio
async def test_unanswerable_query_has_no_fabricated_citations(tmp_path: Path):
    store = RagStore(tmp_path / "index.sqlite3")
    service = RagService(None, document_store=DocumentStore(tmp_path / "documents"), vector_store=store, embeddings=FakeEmbeddings(), chat=FakeChat())
    result = await service.query(RagQueryRequest(question="Unknown", model="qwen2.5:3b"))
    assert result.citations == []
    assert "could not find" in result.answer.lower()


def test_document_processing_triggers_post_ocr_indexing(tmp_path: Path):
    documents = DocumentStore(tmp_path / "documents")
    source = tmp_path / "source.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\nlocal")
    detail = documents.create("source.png", "image/png", source.stat().st_size, 1, source, OCRConfig())
    service = DocumentService(Settings(), store=documents)
    calls = []

    class FakeProcessor:
        def process(self, document_id):
            saved = documents.get(document_id)
            saved.status = "completed"
            documents.save(saved)

    class FakeRag:
        async def index(self, document_ids):
            calls.extend(document_ids)

    service.processor = FakeProcessor()
    service.rag = FakeRag()
    service._process_and_index(detail.document_id)
    assert calls == [detail.document_id]


def test_rag_routes_validate_and_return_structured_data():
    class FakeRag:
        def indexes(self):
            return [{"document_id": "doc", "status": "indexed"}]

        async def index(self, document_ids, chunk_size, overlap):
            return RagIndexResponse(indexes=[{"document_id": document_ids[0], "status": "indexed"}])

        async def query(self, request):
            return RagQueryResponse(answer="local answer", model=request.model, citations=[], retrieved_count=0)

        def index_detail(self, document_id):
            return {"document_id": document_id, "status": "indexed"}

        def delete_index(self, document_id):
            return True

    app.dependency_overrides[get_rag] = lambda: FakeRag()
    try:
        client = TestClient(app)
        assert client.get("/api/v1/rag/indexes").status_code == 200
        assert client.post("/api/v1/rag/index", json={"document_ids": ["doc"]}).status_code == 200
        assert client.post("/api/v1/rag/query", json={"question": "hello", "model": "qwen2.5:3b"}).status_code == 200
        assert client.post("/api/v1/rag/query", json={"question": "", "model": "qwen2.5:3b"}).status_code == 422
    finally:
        app.dependency_overrides.clear()
