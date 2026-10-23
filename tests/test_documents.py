from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.documents import DocumentDetail, OCRConfig, PageResult
from app.services.document_store import DocumentStore, safe_name
from app.services.document_processor import DocumentProcessor


def test_safe_document_store_round_trip(tmp_path: Path):
    store = DocumentStore(tmp_path)
    source = tmp_path / "source.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\nlocal")
    detail = store.create("..\\unsafe name?.png", "image/png", 13, 1, source, OCRConfig())
    assert detail.original_filename == "unsafe_name_.png"
    assert store.get(detail.document_id).document_id == detail.document_id
    assert store.delete(detail.document_id) is True
    assert store.get(detail.document_id) is None


def test_document_store_rejects_invalid_id(tmp_path: Path):
    store = DocumentStore(tmp_path)
    assert store.get("..\\metadata.json") is None


def test_upload_rejects_unsupported_extension():
    client = TestClient(app)
    response = client.post("/api/v1/documents/upload", files={"file": ("payload.exe", b"MZnot-an-image", "application/octet-stream")})
    assert response.status_code == 400
    assert "supported" in response.json()["detail"]


def test_page_result_preserves_unknown_confidence_and_engine():
    page = PageResult(page_number=1, status="completed", text="hello", engine="PyMuPDF", extraction_method="native_pdf_text")
    assert page.confidence is None
    assert page.engine == "PyMuPDF"


class FakeOCR:
    def __init__(self, engine: str, text: str):
        self.engine = engine
        self.text = text
        self.calls = 0

    def extract(self, image_path, page_number, config):
        self.calls += 1
        return PageResult(page_number=page_number, status="completed", text=self.text, extraction_method="ocr", engine=self.engine)


def test_processor_uses_selected_tesseract_without_replacing_it(tmp_path: Path):
    from app.core.config import Settings

    store = DocumentStore(tmp_path)
    source = tmp_path / "source.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\nlocal")
    detail = store.create("page.png", "image/png", source.stat().st_size, 1, source, OCRConfig(primary_engine="tesseract", enable_fallback=True))
    tesseract = FakeOCR("tesseract", "cpu text")
    glm = FakeOCR("glm-ocr", "vision text")
    processor = DocumentProcessor(store, Settings(), paddle=FakeOCR("paddleocr", "paddle text"), glm=glm, tesseract=tesseract)
    processor.process(detail.document_id)
    saved = store.get(detail.document_id)
    assert saved.status == "completed"
    assert saved.pages[0].engine == "tesseract"
    assert saved.pages[0].text == "cpu text"
    assert tesseract.calls == 1
    assert glm.calls == 0


class FakePDFPage:
    def __init__(self, text):
        self.text = text


class FakePDF:
    def __init__(self, pages):
        self.pages = pages
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def __iter__(self):
        return iter(self.pages)


class FakePDFExtractor:
    version = "1.0-test"
    def __init__(self, pages):
        self.pages = pages
    def open(self, path):
        return FakePDF(self.pages)
    @staticmethod
    def usable_text(page):
        return page.text
    @staticmethod
    def render_page(page, dpi):
        return b"rendered-png"


def test_mixed_pdf_uses_native_text_and_rendered_page_ocr(tmp_path: Path):
    from app.core.config import Settings

    store = DocumentStore(tmp_path)
    source = tmp_path / "mixed.pdf"
    source.write_bytes(b"%PDF-test")
    detail = store.create("mixed.pdf", "application/pdf", source.stat().st_size, 2, source, OCRConfig(primary_engine="tesseract", enable_fallback=False))
    tesseract = FakeOCR("tesseract", "scanned page")
    processor = DocumentProcessor(store, Settings(), tesseract=tesseract, pdf_extractor=FakePDFExtractor([FakePDFPage("A sufficiently long native PDF page text."), FakePDFPage("")]))
    processor.process(detail.document_id)
    saved = store.get(detail.document_id)
    assert saved.status == "completed"
    assert saved.pages[0].extraction_method == "native_pdf_text"
    assert saved.pages[0].page_number == 1
    assert saved.pages[1].engine == "tesseract"
    assert saved.pages[1].page_number == 2
    assert not any(path.name.startswith("ocr-page-") for path in tmp_path.rglob("ocr-page-*"))
