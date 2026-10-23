"""Run a real, local Tesseract OCR smoke test without the backend."""

import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from app.core.config import Settings
from app.providers.tesseract import TesseractProvider
from app.schemas.documents import OCRConfig


def main() -> None:
    expected = "LOCAL OCR TEST 12345"
    path = Path(tempfile.gettempdir()) / "local-ai-tesseract-test.png"
    image = Image.new("RGB", (1600, 260), "white")
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 64)
    ImageDraw.Draw(image).text((50, 80), expected, font=font, fill="black")
    image.save(path)
    try:
        result = TesseractProvider(Settings()).extract(path, 1, OCRConfig(primary_engine="tesseract"))
        print(f"engine={result.engine} version={result.engine_version}")
        print(f"expected={expected!r}")
        print(f"actual={result.text!r}")
        print(f"exact_match={result.text == expected}")
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
