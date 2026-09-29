"""Read-only, paginated access to the existing SQLite manifest; safe during ingestion."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from library_ingestor.errors import ConfigurationError


class Catalog:
    def __init__(self, state_dir: Path) -> None:
        self.path = (state_dir / "state.db").resolve()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if not self.path.exists():
            raise ConfigurationError("No hay un índice local. Ejecuta ingest primero.")
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def stats(self) -> dict[str, Any]:
        with self.connect() as db:
            rows = db.execute("SELECT status, COUNT(*) AS n FROM documents GROUP BY status")
            counts = {str(row["status"]): row["n"] for row in rows}
            chunks = db.execute(
                "SELECT COALESCE(SUM(chunks_created),0) FROM documents WHERE status='completed'"
            ).fetchone()[0]
            return {
                "books": counts.get("completed", 0),
                "chunks": chunks,
                "failed": counts.get("failed", 0),
                "pending_recovery": db.execute("SELECT COUNT(*) FROM journal").fetchone()[0],
            }

    def books(self, query: str, offset: int, limit: int) -> dict[str, Any]:
        pattern = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where = """status='completed' AND (
            COALESCE(json_extract(metadata,'$.title'),'') LIKE ? ESCAPE char(92)
            OR COALESCE(json_extract(metadata,'$.author'),'') LIKE ? ESCAPE char(92)
            OR json_extract(metadata,'$.filename') LIKE ? ESCAPE char(92))"""
        with self.connect() as db:
            args = (pattern, pattern, pattern)
            total = db.execute(f"SELECT COUNT(*) FROM documents WHERE {where}", args).fetchone()[0]
            rows = db.execute(
                f"""SELECT file_hash, metadata, chunks_created, ingested_at
                FROM documents WHERE {where}
                ORDER BY COALESCE(json_extract(metadata,'$.title'),
                                  json_extract(metadata,'$.filename')) COLLATE NOCASE, file_hash
                LIMIT ? OFFSET ?""",
                (*args, limit, offset),
            )
            items = []
            for row in rows:
                meta = json.loads(row["metadata"])
                items.append(
                    {
                        "document_id": row["file_hash"],
                        "title": meta.get("title") or meta["filename"],
                        "author": meta.get("author"),
                        "file_type": meta["file_type"],
                        "language": meta.get("language"),
                        "chunks": row["chunks_created"],
                        "ingested_at": row["ingested_at"],
                    }
                )
            return {"items": items, "total": total, "offset": offset, "limit": limit}

    def readable_hashes(self, hashes: list[str]) -> set[str]:
        """Exclude unfinished documents and hashes involved in a replacement journal.

        Read after Qdrant retrieval in a single SQLite snapshot. A replacement that starts
        afterwards cannot invalidate the already retrieved text of the original file hash.
        """
        if not hashes:
            return set()
        with self.connect() as db:
            db.execute("BEGIN")
            placeholders = ",".join("?" for _ in hashes)
            readable = {
                str(row[0])
                for row in db.execute(
                    f"SELECT file_hash FROM documents WHERE status='completed' "
                    f"AND file_hash IN ({placeholders})",
                    hashes,
                )
            }
            for row in db.execute("SELECT data FROM journal WHERE committed=0"):
                readable.difference_update(json.loads(row[0])["hashes"])
            return readable
