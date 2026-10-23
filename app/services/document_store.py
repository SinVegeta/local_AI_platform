"""Safe JSON-backed local document persistence."""

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.document_settings import DOCUMENT_ROOT
from app.schemas.documents import DocumentDetail, DocumentSummary, OCRConfig, PageResult


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def safe_name(value: str) -> str:
    value = Path(value).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:180] or "document"


class DocumentStore:
    def __init__(self, root: Path = DOCUMENT_ROOT) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, filename: str, media_type: str, size_bytes: int, page_count: int, source_path: Path, config: OCRConfig) -> DocumentDetail:
        document_id = uuid.uuid4().hex
        folder = self._folder(document_id)
        folder.mkdir(parents=True, exist_ok=False)
        target = folder / safe_name(filename)
        shutil.copyfile(source_path, target)
        now = utc_now()
        detail = DocumentDetail(
            document_id=document_id, original_filename=safe_name(filename), media_type=media_type,
            size_bytes=size_bytes, page_count=page_count, status="uploaded", created_at=now,
            updated_at=now, config=config,
            pages=[PageResult(page_number=n, status="pending") for n in range(1, page_count + 1)],
        )
        self.save(detail)
        return detail

    def save(self, detail: DocumentDetail) -> None:
        detail.updated_at = utc_now()
        path = self._folder(detail.document_id) / "metadata.json"
        path.write_text(detail.model_dump_json(indent=2), encoding="utf-8")

    def get(self, document_id: str) -> DocumentDetail | None:
        if not re.fullmatch(r"[0-9a-f]{32}", document_id):
            return None
        path = self._folder(document_id) / "metadata.json"
        if not path.is_file():
            return None
        try:
            return DocumentDetail.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list(self) -> list[DocumentSummary]:
        result: list[DocumentSummary] = []
        for folder in self.root.iterdir():
            if folder.is_dir():
                item = self.get(folder.name)
                if item:
                    result.append(DocumentSummary(**item.model_dump(exclude={"config", "pages", "processing_started_at", "completed_at", "error"})))
        return sorted(result, key=lambda item: item.created_at, reverse=True)

    def source_path(self, document_id: str) -> Path:
        detail = self.get(document_id)
        if not detail:
            raise FileNotFoundError(document_id)
        folder = self._folder(document_id)
        files = [p for p in folder.iterdir() if p.name != "metadata.json" and p.is_file()]
        if len(files) != 1:
            raise FileNotFoundError(document_id)
        return files[0]

    def delete(self, document_id: str) -> bool:
        detail = self.get(document_id)
        if not detail:
            return False
        shutil.rmtree(self._folder(detail.document_id))
        return True

    def _folder(self, document_id: str) -> Path:
        folder = (self.root / document_id).resolve()
        if self.root not in folder.parents:
            raise ValueError("Invalid document path")
        return folder
