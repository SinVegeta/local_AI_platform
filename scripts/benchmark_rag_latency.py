"""Sequential local RAG latency benchmark; never downloads models or leaves the machine."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


QUESTIONS = [
    {"id": "factual", "question": "What is this document about?", "keywords": ["document", "system"]},
    {"id": "multi_passage", "question": "Which OCR engines or OCR approaches are discussed?", "keywords": ["tesseract", "paddleocr"]},
    {"id": "concise_summary", "question": "Summarize the document in one short sentence.", "keywords": ["document"]},
    {"id": "unanswerable", "question": "What is the exact color of the author's office chair?", "keywords": [], "expect_no_citations": True},
    {"id": "page_citation", "question": "What does the document say about OCR on page 5?", "keywords": ["ocr"], "expected_page": 5},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--models", default="qwen3:4b,granite4.1:3b,qwen2.5:3b")
    parser.add_argument("--output", type=Path, default=Path("data/rag/latency-benchmark.json"))
    parser.add_argument("--max-questions", type=int, default=5)
    args = parser.parse_args()
    questions = QUESTIONS[:max(1, min(args.max_questions, len(QUESTIONS)))]
    records = []
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=180.0) as client:
        for model in [value.strip() for value in args.models.split(",") if value.strip()]:
            variants = [("default", None), ("thinking_disabled", False)] if model.lower().split(":", 1)[0] == "qwen3" else [("default", None)]
            for variant, think in variants:
                for item in questions:
                    started = time.perf_counter()
                    payload = {"question": item["question"], "model": model, "document_ids": [args.document_id], "top_k": 5}
                    if think is not None:
                        payload["think"] = think
                    try:
                        response = client.post("/api/v1/rag/query", json=payload)
                        data = response.json()
                        elapsed_ms = round((time.perf_counter() - started) * 1000)
                        citations = data.get("citations", []) if response.is_success else []
                        answer = data.get("answer", "") if response.is_success else ""
                        answer_lower = answer.lower()
                        keyword_match = all(keyword.lower() in answer_lower for keyword in item.get("keywords", [])) if item.get("keywords") else None
                        citation_correct = any(citation.get("page_number") == item["expected_page"] for citation in citations) if item.get("expected_page") else None
                        grounding_ok = not citations if item.get("expect_no_citations") else (keyword_match is not False and bool(citations))
                        records.append({"model": model, "variant": variant, "question_id": item["id"], "status_code": response.status_code, "elapsed_ms": elapsed_ms, "total_ms": data.get("total_ms"), "retrieval_ms": data.get("retrieval_ms"), "generation_ms": data.get("generation_ms"), "ollama_total_ms": data.get("ollama_total_ms"), "model_load_ms": data.get("model_load_ms"), "prompt_eval_ms": data.get("prompt_eval_ms"), "generation_eval_ms": data.get("generation_eval_ms"), "prompt_tokens": data.get("prompt_tokens"), "generated_tokens": data.get("generated_tokens"), "tokens_per_second": data.get("tokens_per_second"), "retrieved_count": data.get("retrieved_count", 0), "citation_pages": [citation.get("page_number") for citation in citations], "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest() if answer else None, "keyword_match": keyword_match, "citation_correct": citation_correct, "grounding_check": grounding_ok, "error": None if response.is_success else data.get("detail", "request failed")})
                    except (httpx.HTTPError, ValueError) as exc:
                        records.append({"model": model, "variant": variant, "question_id": item["id"], "status_code": None, "elapsed_ms": round((time.perf_counter() - started) * 1000), "error": str(exc)})
    payload = {"version": "1.0", "started_at": datetime.now(timezone.utc).isoformat(), "document_id": args.document_id, "questions": questions, "models": args.models.split(","), "records": records, "notes": ["Models were run sequentially.", "Answers are stored as SHA-256 hashes to avoid copying document-derived text into the benchmark report.", "Keyword, citation-page, and no-citation checks are basic automated signals and require human review."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
