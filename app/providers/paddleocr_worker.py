"""Worker entry point used by the isolated PaddleOCR environment."""

import json
import sys
from pathlib import Path


def main() -> int:
    request = json.loads(sys.stdin.read())
    image_path = Path(request["image_path"])
    try:
        from paddleocr import PaddleOCR
        import paddle
        device = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
        ocr = PaddleOCR(
            lang=request.get("language", "en"),
            ocr_version="PP-OCRv5",
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        results = ocr.predict(str(image_path))
        blocks = []
        for result in results:
            data = result.json if hasattr(result, "json") else {}
            if callable(data):
                data = data()
            data = data if isinstance(data, dict) else {}
            for text, score, box in zip(data.get("rec_texts", []), data.get("rec_scores", []), data.get("rec_boxes", [])):
                blocks.append({"text": str(text), "confidence": float(score), "bbox": [float(x) for x in box]})
        text = "\n".join(block["text"] for block in blocks).strip()
        print(json.dumps({"ok": True, "device": device, "engine_version": "PaddleOCR PP-OCRv5", "text": text, "blocks": blocks}))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)[:500]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
