"""Live API RAG smoke test using temporary local data and installed Ollama models."""

import time
from pathlib import Path
import httpx
from PIL import Image, ImageDraw


def main() -> None:
    image = Path("data/rag-api-smoke.png")
    image.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1200, 300), "white")
    ImageDraw.Draw(canvas).text((60, 130), "The local retention policy is seven years.", fill="black")
    canvas.save(image)
    base = "http://127.0.0.1:8000"
    document_id = None
    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(f"{base}/api/v1/documents/upload", files={"file": ("rag-api-smoke.png", image.read_bytes(), "image/png")}, data={"primary_engine": "tesseract", "language": "en", "enable_fallback": "false"})
            response.raise_for_status()
            document_id = response.json()["document_id"]
            for _ in range(60):
                detail = client.get(f"{base}/api/v1/documents/{document_id}").json()
                if detail["status"] not in {"queued", "processing"}:
                    break
                time.sleep(0.5)
            assert detail["status"] == "completed", detail
            indexed = client.post(f"{base}/api/v1/rag/index", json={"document_ids": [document_id]}).json()
            answer = client.post(f"{base}/api/v1/rag/query", json={"question": "What is the retention policy?", "model": "qwen2.5:3b", "document_ids": [document_id], "top_k": 3}).json()
            print({"document_id": document_id, "index_status": indexed["indexes"][0]["status"], "answer": answer["answer"], "citations": answer["citations"]})
    finally:
        if document_id:
            try:
                httpx.delete(f"{base}/api/v1/documents/{document_id}", timeout=30.0)
            except httpx.HTTPError:
                pass
        image.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
