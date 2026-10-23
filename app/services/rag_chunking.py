"""Deterministic page-aware text chunking for local retrieval."""

import hashlib
import re


def chunk_pages(document_id: str, pages: list[dict], chunk_size: int = 900, overlap: int = 120) -> list[dict]:
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be at least 100 and overlap must be smaller than chunk_size")
    chunks = []
    for page in pages:
        text = re.sub(r"\s+", " ", (page.get("text") or "")).strip()
        if not text:
            continue
        start = 0
        ordinal = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + (2 if text[boundary:boundary + 2] == ". " else 1)
            content = text[start:end].strip()
            if content:
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                chunks.append({
                    "chunk_id": f"{document_id}-p{page['page_number']}-{ordinal}-{digest[:12]}",
                    "document_id": document_id,
                    "page_number": page["page_number"],
                    "text": content,
                    "content_hash": digest,
                    "extraction_method": page.get("extraction_method"),
                    "engine": page.get("engine"),
                })
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
            ordinal += 1
    return chunks
