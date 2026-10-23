"""Real local PDF integration tests: PyMuPDF rendering + Tesseract OCR."""

from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from app.core.config import Settings
from app.providers.tesseract import TesseractProvider
from app.schemas.documents import OCRConfig
from app.services.document_processor import DocumentProcessor
from app.services.document_store import DocumentStore
from app.services.pdf_extraction import PDFExtractor, PDFProcessingError


def _font():
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 64)


def _scan_png(text: str) -> bytes:
    image = Image.new("RGB", (1600, 260), "white")
    ImageDraw.Draw(image).text((50, 80), text, font=_font(), fill="black")
    output = Path("scan-fixture.png")
    image.save(output)
    try:
        return output.read_bytes()
    finally:
        output.unlink(missing_ok=True)


def _digital_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 100), text, fontsize=18)
    document.save(path)
    document.close()


def _scanned_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page(width=600, height=180)
        page.insert_image(page.rect, stream=_scan_png(text))
    document.save(path)
    document.close()


def _process(path: Path, root: Path):
    store = DocumentStore(root)
    source = root / path.name
    source.write_bytes(path.read_bytes())
    detail = store.create(path.name, "application/pdf", source.stat().st_size, len(pymupdf.open(source)), source, OCRConfig(primary_engine="tesseract", enable_fallback=False, render_dpi=150))
    processor = DocumentProcessor(store, Settings(), tesseract=TesseractProvider(Settings()))
    processor.process(detail.document_id)
    return store.get(detail.document_id), root / detail.document_id


def test_real_digital_pdf_native_extraction(tmp_path: Path):
    pdf = tmp_path / "digital.pdf"
    _digital_pdf(pdf, ["DIGITAL PAGE ONE TEXT", "DIGITAL PAGE TWO TEXT"])
    detail, _ = _process(pdf, tmp_path / "store")
    assert detail.status == "completed"
    assert [page.page_number for page in detail.pages] == [1, 2]
    assert [page.extraction_method for page in detail.pages] == ["native_pdf_text", "native_pdf_text"]
    assert "DIGITAL PAGE ONE TEXT" in detail.pages[0].text
    assert detail.pages[0].engine == "PyMuPDF"


def test_real_scanned_pdf_renders_and_uses_tesseract(tmp_path: Path):
    pdf = tmp_path / "scanned.pdf"
    _scanned_pdf(pdf, ["SCANNED PAGE TEXT 12345"])
    detail, folder = _process(pdf, tmp_path / "store")
    assert detail.status == "completed"
    assert detail.pages[0].page_number == 1
    assert detail.pages[0].engine == "tesseract"
    assert detail.pages[0].extraction_method == "ocr"
    assert "SCANNED PAGE TEXT 12345" in detail.pages[0].text
    assert not any(folder.glob("ocr-page-*"))


def test_real_mixed_pdf_selects_native_and_ocr_per_page(tmp_path: Path):
    pdf = tmp_path / "mixed.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "MIXED DIGITAL PAGE TEXT", fontsize=18)
    page = document.new_page(width=600, height=180)
    page.insert_image(page.rect, stream=_scan_png("MIXED SCANNED PAGE 67890"))
    document.save(pdf)
    document.close()
    detail, folder = _process(pdf, tmp_path / "store")
    assert detail.status == "completed"
    assert detail.pages[0].engine == "PyMuPDF"
    assert detail.pages[0].extraction_method == "native_pdf_text"
    assert detail.pages[1].engine == "tesseract"
    assert detail.pages[1].extraction_method == "ocr"
    assert "MIXED SCANNED PAGE 67890" in detail.pages[1].text
    assert not any(folder.glob("ocr-page-*"))


def test_real_corrupt_pdf_is_rejected(tmp_path: Path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-not-a-real-document")
    try:
        PDFExtractor().open(corrupt)
    except PDFProcessingError as exc:
        assert "malformed" in str(exc) or "could not be opened" in str(exc)
    else:
        raise AssertionError("corrupt PDF unexpectedly opened")


def test_real_encrypted_pdf_is_rejected(tmp_path: Path):
    plain = tmp_path / "plain.pdf"
    encrypted = tmp_path / "encrypted.pdf"
    _digital_pdf(plain, ["PRIVATE CONTENT"])
    document = pymupdf.open(plain)
    document.save(encrypted, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    document.close()
    try:
        PDFExtractor().open(encrypted)
    except PDFProcessingError as exc:
        assert "Encrypted" in str(exc)
    else:
        raise AssertionError("encrypted PDF unexpectedly opened")
