from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.config import Settings
from app.providers.ocr_base import OCRProviderError
from app.providers.tesseract import TesseractProvider
from app.schemas.documents import OCRConfig


def result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_tesseract_success_uses_eng_and_normalizes_metadata(tmp_path: Path):
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    provider = TesseractProvider(Settings(), executable="tesseract-test")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        if "--list-langs" in args:
            return result("List of available languages in \"x\":\neng\nosd\n")
        if "--version" in args:
            return result("tesseract 5.5.3\n")
        return result("Hello local OCR\n")

    with patch("app.providers.tesseract.subprocess.run", side_effect=run):
        page = provider.extract(image, 3, OCRConfig(primary_engine="tesseract"))
    assert page.text == "Hello local OCR"
    assert page.page_number == 3
    assert page.engine == "tesseract"
    assert page.extraction_method == "ocr"
    ocr_args = calls[-1][0]
    assert "-l" in ocr_args and ocr_args[ocr_args.index("-l") + 1] == "eng"
    assert "shell" not in calls[-1][1]


def test_tesseract_rejects_unsupported_language():
    provider = TesseractProvider(Settings(), executable="tesseract-test")
    with patch("app.providers.tesseract.subprocess.run", return_value=result("List of available languages in \"x\":\neng\n")):
        with pytest.raises(OCRProviderError, match="not installed"):
            provider.extract(Path("page.png"), 1, OCRConfig(primary_engine="tesseract", language="fra"))


@pytest.mark.parametrize("failure, message", [
    (FileNotFoundError(), "not found"),
    (subprocess.TimeoutExpired("tesseract", 1), "timed out"),
])
def test_tesseract_missing_or_startup_failure(failure, message):
    provider = TesseractProvider(Settings(), executable="tesseract-test")
    with patch("app.providers.tesseract.subprocess.run", side_effect=failure):
        with pytest.raises(OCRProviderError):
            provider.installed_languages()


def test_tesseract_timeout_and_nonzero_exit(tmp_path: Path):
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    provider = TesseractProvider(Settings(), executable="tesseract-test")
    with patch("app.providers.tesseract.subprocess.run", side_effect=[result("List of available languages in \"x\":\neng\n"), TimeoutError()]):
        with pytest.raises(OCRProviderError):
            provider.extract(image, 1, OCRConfig(primary_engine="tesseract"))
    provider = TesseractProvider(Settings(), executable="tesseract-test")
    with patch("app.providers.tesseract.subprocess.run", side_effect=[result("List of available languages in \"x\":\neng\n"), result("tesseract 5\n"), result("", "bad image", 1)]):
        with pytest.raises(OCRProviderError, match="process"):
            provider.extract(image, 1, OCRConfig(primary_engine="tesseract"))
