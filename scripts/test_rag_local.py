"""Real local RAG smoke test; uses only Ollama models already installed."""

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.providers.embeddings import OllamaEmbeddingProvider
from app.providers.ollama import OllamaProvider
from app.schemas.documents import OCRConfig, PageResult
from app.schemas.rag import RagQueryRequest
from app.services.document_store import DocumentStore
from app.services.rag_service import RagService
from app.services.rag_store import RagStore


async def main() -> None:
    settings = Settings()
    with tempfile.TemporaryDirectory(prefix="local-rag-smoke-") as directory:
        root = Path(directory)
        documents = DocumentStore(root / "documents")
        source = root / "known.png"
        source.write_bytes(b"\x89PNG\r\n\x1a\nlocal-smoke")
        detail = documents.create("known.png", "image/png", source.stat().st_size, 2, source, OCRConfig())
        detail.status = "completed"
        detail.pages = [
            PageResult(page_number=1, status="completed", text="The local platform stores invoices in the finance archive.", extraction_method="native_pdf_text", engine="PyMuPDF"),
            PageResult(page_number=2, status="completed", text="The retention period for invoices is seven years.", extraction_method="ocr", engine="tesseract"),
        ]
        documents.save(detail)
        service = RagService(settings, document_store=documents, vector_store=RagStore(root / "index.sqlite3"), embeddings=OllamaEmbeddingProvider(settings), chat=OllamaProvider(settings))
        indexed = await service.index([detail.document_id])
        dimension = indexed.indexes[0]["dimension"]
        result = await service.query(RagQueryRequest(question="How long are invoices retained?", model="qwen2.5:3b", document_ids=[detail.document_id], top_k=3))
        print({"document_id": detail.document_id, "embedding_dimension": dimension, "retrieved_count": result.retrieved_count, "answer": result.answer, "citations": [citation.model_dump() for citation in result.citations]})


if __name__ == "__main__":
    asyncio.run(main())
