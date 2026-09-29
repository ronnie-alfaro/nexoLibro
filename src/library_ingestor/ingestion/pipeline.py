"""Sequential ingestion with disk-spooled vectors and recoverable Qdrant replacement."""

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from itertools import batched
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from library_ingestor.chunking.chunker import StructuralChunker
from library_ingestor.config import Settings
from library_ingestor.discovery import discover
from library_ingestor.embeddings.embedder import Embedder
from library_ingestor.errors import ConfigurationError, RecoveryError, StoreError
from library_ingestor.ingestion.state import StateDB, now
from library_ingestor.metadata import base_metadata
from library_ingestor.models import DocumentMetadata, IngestionResult, ParsedDocument
from library_ingestor.parsers.base import ParserRegistry
from library_ingestor.utils.hashing import file_sha256
from library_ingestor.vectorstore.qdrant_store import QdrantStore, make_point

logger = logging.getLogger(__name__)
Progress = Callable[[str], None]


def index_identity(settings: Settings) -> dict[str, object]:
    return {
        "schema": 1,
        "model": settings.embedding.model,
        "revision": settings.embedding.revision,
        "normalize": settings.embedding.normalize,
        "chunking": settings.chunking.model_dump(),
        "parsing": settings.parsing.model_dump(mode="json"),
        "state_dir": str(settings.ingestion.state_dir.resolve()),
        "host": settings.qdrant.host,
        "port": settings.qdrant.port,
        "collection": settings.qdrant.collection,
    }


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        parsers: ParserRegistry,
        embedder: Embedder,
        store: QdrantStore,
        state: StateDB,
        progress: Progress = print,
    ) -> None:
        self.settings = settings
        self.parsers = parsers
        self.embedder = embedder
        self.store = store
        self.state = state
        self.progress = progress
        self._ready = False
        self._seen_hashes: set[str] = set()

    def _ensure(self) -> None:
        if not self._ready:
            special_tokens = len(self.embedder.tokenizer.encode("", add_special_tokens=True))
            if self.settings.chunking.max_tokens + special_tokens > self.embedder.max_tokens:
                raise ConfigurationError("Chunk max_tokens exceeds embedding model context window")
            self.store.ensure_collection(self.embedder.dimension, index_identity(self.settings))
            self._ready = True

    def recover(self) -> None:
        """Rollback interrupted writes, or finish cleanup after a committed SQLite transaction."""
        for journal in self.state.journals():
            data = json.loads(journal["data"])
            directory = Path(data["directory"])
            try:
                if not journal["committed"]:
                    backup = directory / "old.jsonl"
                    if not backup.exists():
                        raise RecoveryError(f"Missing rollback data: {backup}")
                    self.store.delete(data["hashes"])
                    self.store.upload(backup)
                self.state.remove_journal(journal["id"])
                shutil.rmtree(directory, ignore_errors=True)
            except Exception as exc:
                raise RecoveryError("Recovery failed; keep state/spool and restore Qdrant") from exc
        self.state.interrupted()
        # No journal means no Qdrant mutation could have begun for this spool.
        spool = self.settings.ingestion.state_dir / "spool"
        if spool.exists():
            for child in spool.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)

    def _prepare(self, document: ParsedDocument, directory: Path) -> int:
        chunker = StructuralChunker(self.settings.chunking, self.embedder.tokenizer)
        count = 0
        indexed_at = now()
        with (directory / "new.jsonl").open("w") as out:
            for batch in batched(chunker.chunks(document), self.settings.embedding.batch_size):
                vectors = self.embedder.encode([chunk.text for chunk in batch])
                if len(vectors) != len(batch):
                    raise StoreError("Embedding count differs from chunk count")
                for chunk, vector in zip(batch, vectors, strict=True):
                    if len(vector) != self.embedder.dimension:
                        raise StoreError("Inconsistent embedding dimension")
                    point = make_point(
                        chunk, document.metadata, vector, self.settings.embedding.model, indexed_at
                    )
                    out.write(point.model_dump_json() + "\n")
                    count += 1
                self.progress(f"Embedding: {count} chunks prepared")
            out.flush()
            os.fsync(out.fileno())
        if not count:
            raise StoreError("Refusing to index a document without chunks")
        return count

    def _replace(
        self,
        metadata: DocumentMetadata,
        mtime: float,
        count: int,
        directory: Path,
        retired: str | None,
    ) -> None:
        hashes = list(dict.fromkeys([metadata.file_hash] + ([retired] if retired else [])))
        self.store.backup(hashes, directory / "old.jsonl")
        # fsync directory entries before making the durable journal visible.
        for target in (directory, directory.parent):
            fd = os.open(target, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        transaction = directory.name
        self.state.add_journal(transaction, {"directory": str(directory), "hashes": hashes})
        try:
            self.store.delete(hashes)
            self.store.upload(directory / "new.jsonl")
            if self.store.count(metadata.file_hash) != count:
                raise StoreError("Qdrant point count did not match prepared chunks")
            self.state.complete(
                metadata, mtime, count, self.settings.embedding.model, transaction, retired
            )
        except BaseException:
            # Includes Ctrl-C; a hard kill is recovered from the journal on next ingest.
            self.recover()
            raise
        self.state.remove_journal(transaction)
        shutil.rmtree(directory, ignore_errors=True)

    def _process(self, path: Path, root: Path, force: bool) -> tuple[str, int]:
        digest = file_sha256(path)
        metadata = base_metadata(path, root, digest)
        mtime = path.stat().st_mtime
        previous = self.state.document(digest)
        retired = self.state.retired_hash(str(path), digest)
        if (
            previous
            and previous["status"] == "completed"
            and (not force or digest in self._seen_hashes)
            and retired is None
        ):
            if self.store.count(digest) == previous["chunks_created"]:
                self.state.alias(digest, str(path))
                self._seen_hashes.add(digest)
                return "skipped", 0
        self.state.start(metadata, mtime, self.settings.embedding.model)
        directory = self.settings.ingestion.state_dir / "spool" / str(uuid4())
        directory.mkdir(parents=True)
        try:
            self.progress("Parsing…")
            document = self.parsers.get(path).parse(path, metadata)
            self._ensure()
            count = self._prepare(document, directory)
            # Detect edits during parsing/encoding before replacing anything in Qdrant.
            if file_sha256(path) != digest:
                raise StoreError("Source changed during ingestion; retry when it is stable")
            self._replace(document.metadata, mtime, count, directory, retired)
            self._seen_hashes.add(digest)
            return "completed", count
        except RecoveryError:
            raise  # Never continue modifying a store whose rollback is incomplete.
        except Exception as exc:
            self.state.fail(digest, str(path), str(exc))
            if not self.state.journals():
                shutil.rmtree(directory, ignore_errors=True)
            raise

    def ingest(self, root: Path, recursive: bool = True, force: bool = False) -> IngestionResult:
        started = time.monotonic()
        result = IngestionResult()
        root = root.resolve()
        self._seen_hashes.clear()
        self._ready = False
        base = root if root.is_dir() else root.parent
        with FileLock(self.settings.ingestion.state_dir / "ingestion.lock", timeout=0):
            self.state.bind(index_identity(self.settings))
            existing = self.store.identity()
            if existing is not None and existing != index_identity(self.settings):
                raise ConfigurationError("Collection identity differs from this configuration")
            self.recover()
            self.progress("Discovering library…")
            # Only pathnames in memory, never the library's contents.
            files = list(discover(root, self.settings.ingestion.supported_extensions, recursive))
            result.discovered = len(files)
            self.progress(f"Found {len(files)} documents")
            for index, path in enumerate(files, 1):
                self.progress(f"[{index:03d}/{len(files)}] {path.name}")
                try:
                    status, count = self._process(path, base, force)
                    if status == "skipped":
                        result.skipped += 1
                        self.progress("→ Skipped: already indexed")
                    else:
                        result.processed += 1
                        result.chunks_created += count
                        self.progress(f"✓ Completed — Stored: {count}")
                except (RecoveryError, ConfigurationError):
                    raise
                except Exception as exc:
                    logger.exception("Document failed: %s", path)
                    self.state.record_attempt(None, str(path), "failed", str(exc))
                    result.failed += 1
                    result.failures.append((str(path), str(exc)))
                    self.progress(f"✗ Failed: {path.name}: {exc}")
        result.elapsed_seconds = time.monotonic() - started
        return result
