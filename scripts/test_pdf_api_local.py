"""Exercise one real scanned PDF through the running local API."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pymupdf
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    expected = "API SCANNED PDF TEST 24680"
    image = Image.new("RGB", (1600, 260), "white")
    ImageDraw.Draw(image).text((50, 80), expected, font=ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 64), fill="black")
    image_path = Path("api-pdf-scan.png")
    pdf_path = Path("api-pdf-scan.pdf")
    image.save(image_path)
    document = pymupdf.open()
    page = document.new_page(width=600, height=180)
    page.insert_image(page.rect, filename=str(image_path))
    document.save(pdf_path)
    document.close()
    client = httpx.Client(timeout=180)
    try:
        response = client.post("http://127.0.0.1:8000/api/v1/documents/upload", files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")}, data={"primary_engine": "tesseract", "language": "en", "enable_fallback": "false"})
        response.raise_for_status()
        document_id = response.json()["document_id"]
        detail = None
        for _ in range(30):
            time.sleep(1)
            detail = client.get(f"http://127.0.0.1:8000/api/v1/documents/{document_id}").json()
            if detail["status"] not in {"queued", "processing"}:
                break
        page_result = detail["pages"][0]
        print({"status": detail["status"], "engine": page_result["engine"], "text": page_result["text"], "exact_match": expected in page_result["text"]})
        client.delete(f"http://127.0.0.1:8000/api/v1/documents/{document_id}")
    finally:
        image_path.unlink(missing_ok=True)
        pdf_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
