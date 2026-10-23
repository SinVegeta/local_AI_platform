"""PyMuPDF-backed native extraction and single-page rendering."""

from pathlib import Path


class PDFProcessingError(RuntimeError):
    pass


class PDFDependencyError(PDFProcessingError):
    pass


class PDFExtractor:
    def __init__(self) -> None:
        try:
            import pymupdf
        except ImportError as exc:
            raise PDFDependencyError("PyMuPDF is not installed in the backend environment") from exc
        self.pymupdf = pymupdf
        self.version = pymupdf.VersionBind

    def open(self, path: Path):
        try:
            document = self.pymupdf.open(path)
            if document.needs_pass:
                document.close()
                raise PDFProcessingError("Encrypted PDFs are not supported")
            if len(document) == 0:
                document.close()
                raise PDFProcessingError("The PDF contains no pages")
            return document
        except PDFProcessingError:
            raise
        except Exception as exc:
            raise PDFProcessingError("The PDF is malformed or could not be opened") from exc

    @staticmethod
    def usable_text(page) -> str:
        try:
            return page.get_text("text").strip()
        except Exception as exc:
            raise PDFProcessingError("A PDF page's text could not be extracted") from exc

    def render_page(self, page, dpi: int) -> bytes:
        try:
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            return pixmap.tobytes("png")
        except Exception as exc:
            raise PDFProcessingError("A PDF page could not be rendered") from exc
