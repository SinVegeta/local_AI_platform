"""SQLite-backed persistent vector storage for modest local collections."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path


class RagStore:
    def __init__(self, path: Path = Path("data/rag/index.sqlite3")) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS indexes (
                    document_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    embedding_model TEXT,
                    dimension INTEGER,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    source_hash TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    extraction_method TEXT,
                    engine TEXT,
                    FOREIGN KEY(document_id) REFERENCES indexes(document_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
            """)

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def get(self, document_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM indexes WHERE document_id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM indexes ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def replace(self, document_id: str, status: str, model: str, dimension: int, source_hash: str, chunks: list[dict], error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO indexes(document_id,status,embedding_model,dimension,chunk_count,source_hash,error,updated_at) VALUES(?,?,?,?,?,?,?,datetime('now')) ON CONFLICT(document_id) DO UPDATE SET status=excluded.status,embedding_model=excluded.embedding_model,dimension=excluded.dimension,chunk_count=excluded.chunk_count,source_hash=excluded.source_hash,error=excluded.error,updated_at=excluded.updated_at", (document_id, status, model, dimension, len(chunks), source_hash, error))
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.executemany("INSERT INTO chunks(chunk_id,document_id,page_number,text,embedding,dimension,content_hash,extraction_method,engine) VALUES(?,?,?,?,?,?,?,?,?)", [(item["chunk_id"], document_id, item["page_number"], item["text"], json.dumps(item["embedding"]), dimension, item["content_hash"], item.get("extraction_method"), item.get("engine")) for item in chunks])

    def mark(self, document_id: str, status: str, model: str, source_hash: str, error: str | None = None) -> None:
        existing = self.get(document_id)
        self.replace(document_id, status, model, existing.get("dimension", 0) if existing else 0, source_hash, [], error)

    def delete(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM indexes WHERE document_id = ?", (document_id,))
        return cursor.rowcount > 0

    def search(self, query_vector: list[float], document_ids: list[str] | None, top_k: int) -> list[dict]:
        with self._connect() as connection:
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                rows = connection.execute(f"SELECT c.*, i.status, i.embedding_model FROM chunks c JOIN indexes i ON i.document_id=c.document_id WHERE i.status='indexed' AND c.document_id IN ({placeholders})", document_ids).fetchall()
            else:
                rows = connection.execute("SELECT c.*, i.status, i.embedding_model FROM chunks c JOIN indexes i ON i.document_id=c.document_id WHERE i.status='indexed'").fetchall()
        results = []
        for row in rows:
            vector = json.loads(row["embedding"])
            if len(vector) != len(query_vector):
                continue
            score = sum(a * b for a, b in zip(query_vector, vector)) / max(math.sqrt(sum(a * a for a in query_vector)) * math.sqrt(sum(a * a for a in vector)), 1e-12)
            item = dict(row)
            item["score"] = score
            results.append(item)
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
