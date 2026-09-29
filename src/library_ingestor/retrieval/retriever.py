"""Bounded ANN retrieval with embedding reuse, overlap removal and book diversity."""

import threading
from collections import OrderedDict
from typing import Any

from qdrant_client import models

from library_ingestor.config import Settings
from library_ingestor.embeddings.embedder import Embedder
from library_ingestor.errors import ConfigurationError
from library_ingestor.ingestion.pipeline import index_identity
from library_ingestor.retrieval.catalog import Catalog
from library_ingestor.retrieval.models import Source
from library_ingestor.vectorstore.qdrant_store import QdrantStore


class Retriever:
    def __init__(
        self, settings: Settings, embedder: Embedder, store: QdrantStore, catalog: Catalog
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.catalog = catalog
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def retrieve(self, query: str, document_id: str | None = None) -> list[Source]:
        # Encoding is intentionally serialized: one reusable BGE model, bounded accelerator RAM.
        with self._lock:
            if self.store.identity() != index_identity(self.settings):
                raise ConfigurationError("La configuración no coincide con el índice existente.")
            key = query.strip()
            if key in self._cache:
                vector = self._cache[key]
                self._cache.move_to_end(key)
            else:
                vector = self.embedder.encode([key])[0]
                capacity = self.settings.retrieval.embedding_cache_size
                if capacity:
                    self._cache[key] = vector
                    while len(self._cache) > capacity:
                        self._cache.popitem(last=False)
            scope = (
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
                if document_id
                else None
            )
            hits = self.store.client.query_points(
                self.settings.qdrant.collection,
                query=vector,
                limit=self.settings.retrieval.candidate_limit,
                score_threshold=self.settings.retrieval.min_score,
                query_filter=scope,
                with_payload=True,
                with_vectors=False,
            ).points
            hashes = [str((point.payload or {}).get("file_hash", "")) for point in hits]
            readable = self.catalog.readable_hashes(hashes)
            return self._select(hits, readable, document_id)

    def _select(
        self, hits: list[models.ScoredPoint], readable: set[str], scoped_document: str | None
    ) -> list[Source]:
        selected: list[Source] = []
        word_sets: list[set[str]] = []
        per_book: dict[str, int] = {}
        options = self.settings.retrieval
        for point in hits:
            payload: dict[str, Any] = point.payload or {}
            digest = str(payload.get("file_hash", ""))
            text = str(payload.get("text", "")).strip()
            if digest not in readable or not text:
                continue
            cap = options.source_limit if scoped_document else options.max_per_book
            if per_book.get(digest, 0) >= cap:
                continue
            words = set(text.casefold().split())
            if any(len(words & seen) / max(1, len(words | seen)) > 0.82 for seen in word_sets):
                continue
            selected.append(
                Source(
                    number=len(selected) + 1,
                    point_id=str(point.id),
                    document_id=digest,
                    title=payload.get("title") or payload.get("filename") or "Sin título",
                    author=payload.get("author"),
                    filename=payload.get("filename", ""),
                    section=payload.get("section") or payload.get("heading"),
                    page_number=payload.get("page_number"),
                    chunk_index=payload.get("chunk_index", 0),
                    score=point.score,
                    text=text,
                )
            )
            word_sets.append(words)
            per_book[digest] = per_book.get(digest, 0) + 1
            if len(selected) == options.source_limit:
                break
        return selected
