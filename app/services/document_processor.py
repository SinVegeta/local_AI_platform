"""Sequential, bounded PDF/image OCR orchestration."""

import time
import tempfile
from pathlib import Path

from app.providers.glm_ocr import GLMOCRProvider
from app.providers.ocr_base import OCRProviderError
from app.providers.paddleocr import PaddleOCRProvider
from app.providers.tesseract import TesseractProvider
from app.schemas.documents import DocumentDetail, PageResult
from app.services.document_store import DocumentStore, utc_now
from app.services.pdf_extraction import PDFDependencyError, PDFExtractor, PDFProcessingError


class DocumentProcessor:
    def __init__(self, store: DocumentStore, settings, paddle=None, glm=None, tesseract=None, pdf_extractor=None) -> None:
        self.store = store
        self.settings = settings
        self.paddle = paddle or PaddleOCRProvider()
        self.glm = glm or GLMOCRProvider(settings)
        self.tesseract = tesseract or TesseractProvider(settings)
        self.pdf_extractor = pdf_extractor

    def process(self, document_id: str) -> None:
        detail = self.store.get(document_id)
        if not detail:
            return
        detail.status = "processing"
        detail.processing_started_at = utc_now()
        self.store.save(detail)
        try:
            if detail.media_type == "application/pdf":
                self._process_pdf(detail)
            else:
                self._process_image(detail)
            failed = sum(page.status == "failed" for page in detail.pages)
            detail.status = "partially_completed" if failed and failed < len(detail.pages) else "failed" if failed else "completed"
        except (PDFDependencyError, PDFProcessingError) as exc:
            detail.status = "failed"
            detail.error = str(exc)
            for page in detail.pages:
                if page.status == "pending":
                    page.status = "failed"
                    page.error = str(exc)
        except Exception:
            detail.status = "failed"
            detail.error = "Document processing failed. Check the per-page errors."
        detail.completed_at = utc_now()
        self.store.save(detail)

    def _process_pdf(self, detail: DocumentDetail) -> None:
        source = self.store.source_path(detail.document_id)
        extractor = self.pdf_extractor or PDFExtractor()
        with extractor.open(source) as pdf:
            for index, page in enumerate(pdf):
                try:
                    result = extractor.usable_text(page)
                    if result and not detail.config.force_ocr and len(result) >= 20:
                        detail.pages[index] = PageResult(page_number=index + 1, status="completed", text=result, extraction_method="native_pdf_text", engine="PyMuPDF", engine_version=extractor.version)
                    else:
                        with tempfile.TemporaryDirectory(prefix="ocr-page-", dir=str(self.store.root / detail.document_id)) as work_dir:
                            image_path = Path(work_dir) / f"page-{index + 1}.png"
                            image_path.write_bytes(extractor.render_page(page, min(detail.config.render_dpi, self.settings.max_render_dpi)))
                            detail.pages[index] = self._ocr_page(detail, index + 1, image_path, low_native=not bool(result))
                except (PDFProcessingError, OCRProviderError) as exc:
                    detail.pages[index] = PageResult(page_number=index + 1, status="failed", error=str(exc), warnings=["Other pages may still complete."])
                detail.completed_pages = sum(page.status == "completed" for page in detail.pages)
                self.store.save(detail)

    def _process_image(self, detail: DocumentDetail) -> None:
        detail.pages[0] = self._ocr_page(detail, 1, self.store.source_path(detail.document_id))
        detail.completed_pages = sum(page.status == "completed" for page in detail.pages)
        self.store.save(detail)

    def _ocr_page(self, detail: DocumentDetail, page_number: int, image_path: Path, low_native: bool = False) -> PageResult:
        start = time.perf_counter()
        if detail.config.primary_engine == "tesseract":
            try:
                result = self.tesseract.extract(image_path, page_number, detail.config)
                result.duration_ms = int((time.perf_counter() - start) * 1000)
                result.selection_reason = "primary_selected"
                if low_native:
                    result.warnings.append("Native PDF text was empty or too short; rendered-page OCR was used.")
                return result
            except OCRProviderError as exc:
                if detail.config.enable_fallback:
                    try:
                        fallback = self.glm.extract(image_path, page_number, detail.config)
                        fallback.primary_text = ""
                        fallback.primary_engine = "tesseract"
                        fallback.selection_reason = "fallback_after_primary_failure"
                        fallback.warnings.append(f"Tesseract failed: {str(exc)}")
                        return fallback
                    except OCRProviderError as fallback_exc:
                        return PageResult(page_number=page_number, status="failed", extraction_method="ocr", engine="tesseract", error=f"Tesseract and GLM-OCR failed: {fallback_exc}")
                return PageResult(page_number=page_number, status="failed", extraction_method="ocr", engine="tesseract", error=str(exc))
        if detail.config.primary_engine == "glm-ocr":
            try:
                result = self.glm.extract(image_path, page_number, detail.config)
                result.duration_ms = int((time.perf_counter() - start) * 1000)
                result.selection_reason = "primary_selected"
                return result
            except OCRProviderError as exc:
                return PageResult(page_number=page_number, status="failed", extraction_method="vision_fallback", engine="glm-ocr", error=str(exc))
        try:
            primary = self.paddle.extract(image_path, page_number, detail.config)
            primary.duration_ms = int((time.perf_counter() - start) * 1000)
            if detail.config.enable_fallback and (not primary.text.strip() or (primary.confidence is not None and primary.confidence < detail.config.fallback_confidence_threshold)):
                try:
                    fallback = self.glm.extract(image_path, page_number, detail.config)
                    primary.fallback_text = fallback.text
                    primary.fallback_engine = fallback.engine
                    primary.primary_text = primary.text
                    primary.primary_engine = primary.engine
                    primary.primary_confidence = primary.confidence
                    primary.selection_reason = "primary_retained_fallback_for_review"
                    primary.warnings.append("GLM-OCR fallback was also run; primary output remains the selected result for review.")
                    primary.warnings.append(f"Fallback text retained in metadata is {len(fallback.text)} characters; outputs are not silently merged.")
                except OCRProviderError as exc:
                    primary.warnings.append(f"GLM-OCR fallback failed: {str(exc)}")
            if low_native:
                primary.warnings.append("Native PDF text was empty or too short; OCR was used.")
            return primary
        except OCRProviderError as exc:
            if not detail.config.enable_fallback:
                return PageResult(page_number=page_number, status="failed", extraction_method="ocr", engine="paddleocr", error=str(exc))
            try:
                fallback = self.glm.extract(image_path, page_number, detail.config)
                fallback.primary_text = ""
                fallback.primary_engine = "paddleocr"
                fallback.selection_reason = "fallback_after_primary_failure"
                fallback.warnings.append(f"Primary PaddleOCR failed: {str(exc)}")
                return fallback
            except OCRProviderError as fallback_exc:
                return PageResult(page_number=page_number, status="failed", error=f"Primary OCR and fallback failed: {fallback_exc}")

