"""SQLite manifest, path aliases and durable transaction journal."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from library_ingestor.errors import ConfigurationError
from library_ingestor.models import DocumentMetadata, Status


def now() -> str:
    return datetime.now(UTC).isoformat()


class StateDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents(
            file_hash TEXT PRIMARY KEY, filepath TEXT NOT NULL, file_size INTEGER NOT NULL,
            modified_time REAL NOT NULL, status TEXT NOT NULL, chunks_created INTEGER DEFAULT 0,
            embedding_model TEXT NOT NULL, ingested_at TEXT, error TEXT, metadata TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS paths(filepath TEXT PRIMARY KEY, file_hash TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS paths_hash ON paths(file_hash);
        CREATE TABLE IF NOT EXISTS attempts(
            id INTEGER PRIMARY KEY, file_hash TEXT, filepath TEXT, status TEXT,
            at TEXT, error TEXT
        );
        CREATE TABLE IF NOT EXISTS journal(
            id TEXT PRIMARY KEY, data TEXT NOT NULL, committed INTEGER NOT NULL DEFAULT 0
        );
        """)

    def close(self) -> None:
        self.connection.close()

    def bind(self, identity: dict[str, Any]) -> None:
        encoded = json.dumps(identity, sort_keys=True)
        row = self.connection.execute("SELECT value FROM settings WHERE key='identity'").fetchone()
        if row and row[0] != encoded:
            raise ConfigurationError(
                "State belongs to a different collection/model/chunking configuration. "
                "Use a new collection and state_dir for a new index."
            )
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO settings VALUES('identity', ?)", (encoded,)
            )

    def document(self, digest: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE file_hash=?", (digest,)
        ).fetchone()
        return dict(row) if row else None

    def path_hash(self, path: str) -> str | None:
        row = self.connection.execute(
            "SELECT file_hash FROM paths WHERE filepath=?", (path,)
        ).fetchone()
        return str(row[0]) if row else None

    def retired_hash(self, path: str, digest: str) -> str | None:
        old = self.path_hash(path)
        if not old or old == digest:
            return None
        count = self.connection.execute(
            "SELECT count(*) FROM paths WHERE file_hash=? AND filepath<>?", (old, path)
        ).fetchone()[0]
        return old if count == 0 else None

    def record_attempt(
        self, digest: str | None, path: str, status: Status, error: str | None = None
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO attempts(file_hash,filepath,status,at,error) VALUES(?,?,?,?,?)",
                (digest, path, status, now(), error),
            )

    def start(self, metadata: DocumentMetadata, mtime: float, model: str) -> None:
        with self.connection:
            # A failed force operation must not invalidate the previous completed version.
            self.connection.execute(
                """
                INSERT INTO documents(file_hash,filepath,file_size,modified_time,status,
                                      embedding_model,metadata)
                VALUES(?,?,?,?,'processing',?,?)
                ON CONFLICT(file_hash) DO UPDATE SET
                  status=CASE WHEN status='completed' THEN status ELSE 'processing' END,
                  error=NULL
            """,
                (
                    metadata.file_hash,
                    metadata.filepath,
                    metadata.file_size,
                    mtime,
                    model,
                    metadata.model_dump_json(),
                ),
            )
        self.record_attempt(metadata.file_hash, metadata.filepath, "processing")

    def fail(self, digest: str, path: str, error: str) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE documents SET
                status=CASE WHEN status='completed' THEN status ELSE 'failed' END, error=?
                WHERE file_hash=?""",
                (error, digest),
            )
        self.record_attempt(digest, path, "failed", error)

    def complete(
        self,
        metadata: DocumentMetadata,
        mtime: float,
        chunks: int,
        model: str,
        transaction_id: str,
        retired: str | None,
    ) -> None:
        with self.connection:
            if retired:
                self.connection.execute("DELETE FROM documents WHERE file_hash=?", (retired,))
            self.connection.execute(
                """INSERT OR REPLACE INTO documents VALUES(
                ?,?,?,?,'completed',?,?,?,NULL,?)""",
                (
                    metadata.file_hash,
                    metadata.filepath,
                    metadata.file_size,
                    mtime,
                    chunks,
                    model,
                    now(),
                    metadata.model_dump_json(),
                ),
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO paths VALUES(?,?)", (metadata.filepath, metadata.file_hash)
            )
            self.connection.execute("UPDATE journal SET committed=1 WHERE id=?", (transaction_id,))

    def alias(self, digest: str, path: str) -> None:
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO paths VALUES(?,?)", (path, digest))
        self.record_attempt(digest, path, "skipped")

    def add_journal(self, transaction_id: str, data: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO journal(id,data) VALUES(?,?)", (transaction_id, json.dumps(data))
            )

    def journals(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM journal")]

    def remove_journal(self, transaction_id: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM journal WHERE id=?", (transaction_id,))

    def books(self, query: str = "") -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM documents ORDER BY filepath")
        return [
            dict(row)
            for row in rows
            if query.casefold() in (row["metadata"] + row["file_hash"]).casefold()
        ]

    def counts(self) -> dict[str, int]:
        return dict(
            self.connection.execute("SELECT status,count(*) FROM documents GROUP BY status")
        )

    def interrupted(self) -> None:
        with self.connection:
            self.connection.execute("""UPDATE documents SET status='failed',
                error='Interrupted; safe to retry' WHERE status IN ('pending','processing')""")
