"""Document application service and bounded local job queue."""

import mimetypes
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.config import Settings
from app.core.document_settings import MAX_PDF_PAGES, MAX_UPLOAD_BYTES
from app.schemas.documents import DocumentDetail, OCRConfig
from app.services.document_processor import DocumentProcessor
from app.services.document_store import DocumentStore
from app.services.pdf_extraction import PDFDependencyError, PDFExtractor, PDFProcessingError
from app.services.rag_service import RagService


class DocumentInputError(ValueError):
    pass


class DocumentBusyError(ValueError):
    pass


class DocumentService:
    def __init__(self, settings: Settings, store: DocumentStore | None = None) -> None:
        self.store = store or DocumentStore()
        self.processor = DocumentProcessor(self.store, settings)
        self.rag = RagService(settings, document_store=self.store)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-ocr")

    async def upload_bytes(self, filename: str, data: bytes, config: OCRConfig) -> DocumentDetail:
        filename = Path(filename or "").name
        extension = Path(filename).suffix.lower()
        media_type = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(extension)
        if not media_type:
            raise DocumentInputError("Only PDF, PNG, JPG, and JPEG files are supported")
        if len(data) > MAX_UPLOAD_BYTES:
            raise DocumentInputError(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit")
        if not self._matches_signature(media_type, data):
            raise DocumentInputError("File content does not match its declared extension")
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp:
            temp.write(data)
            temp_path = Path(temp.name)
        try:
            page_count = self._page_count(temp_path, media_type)
            if page_count > MAX_PDF_PAGES:
                raise DocumentInputError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit")
            detail = self.store.create(filename, media_type, len(data), page_count, temp_path, config)
        finally:
            temp_path.unlink(missing_ok=True)
        detail.status = "queued"
        self.store.save(detail)
        self.executor.submit(self._process_and_index, detail.document_id)
        return detail

    def _process_and_index(self, document_id: str) -> None:
        self.processor.process(document_id)
        detail = self.store.get(document_id)
        if detail and detail.status in {"completed", "partially_completed"}:
            try:
                asyncio.run(self.rag.index([document_id]))
            except Exception:
                # OCR completion remains valid even if the optional embedding
                # runtime is unavailable; the index endpoint can retry later.
                pass

    def delete(self, document_id: str) -> bool:
        detail = self.store.get(document_id)
        if detail and detail.status in {"queued", "processing"}:
            raise DocumentBusyError("Document processing is still active; delete it after completion")
        return self.store.delete(document_id)

    def _page_count(self, path: Path, media_type: str) -> int:
        if media_type != "application/pdf":
            return 1
        try:
            with PDFExtractor().open(path) as pdf:
                return len(pdf)
        except PDFDependencyError as exc:
            raise DocumentInputError(str(exc)) from exc
        except PDFProcessingError as exc:
            raise DocumentInputError(str(exc)) from exc
        except DocumentInputError:
            raise
        except Exception as exc:
            raise DocumentInputError("The PDF is empty or corrupted") from exc

    @staticmethod
    def _matches_signature(media_type: str, data: bytes) -> bool:
        return ((media_type == "application/pdf" and data.startswith(b"%PDF-")) or
                (media_type == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n")) or
                (media_type == "image/jpeg" and data.startswith(b"\xff\xd8\xff")))
