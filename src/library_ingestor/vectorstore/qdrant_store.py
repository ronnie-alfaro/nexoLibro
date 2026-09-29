"""Explicit Qdrant operations; no hosted inference and no implicit embeddings."""

import math
import os
from collections.abc import Iterator
from itertools import batched
from pathlib import Path
from typing import Any, cast

from qdrant_client import QdrantClient, models

from library_ingestor.config import QdrantConfig
from library_ingestor.errors import ConfigurationError, StoreError
from library_ingestor.models import DocumentChunk, DocumentMetadata


def document_filter(hashes: list[str]) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="file_hash",
                match=models.MatchAny(any=hashes),
            )
        ]
    )


def make_point(
    chunk: DocumentChunk,
    metadata: DocumentMetadata,
    vector: list[float],
    embedding_model: str,
    ingested_at: str,
) -> models.PointStruct:
    if not vector or not all(math.isfinite(value) for value in vector):
        raise StoreError("Embedding contains empty/non-finite values")
    return models.PointStruct(
        id=chunk.id,
        vector=vector,
        payload={
            **metadata.model_dump(),
            **chunk.metadata.model_dump(),
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "embedding_model": embedding_model,
            "ingested_at": ingested_at,
        },
    )


def read_points(path: Path) -> Iterator[models.PointStruct]:
    with path.open() as source:
        for line in source:
            yield models.PointStruct.model_validate_json(line)


class QdrantStore:
    def __init__(self, config: QdrantConfig, client: QdrantClient | None = None) -> None:
        self.config = config
        self.client = client or QdrantClient(
            host=config.host,
            port=config.port,
            timeout=config.timeout,
            cloud_inference=False,
            trust_env=False,
        )

    def close(self) -> None:
        self.client.close()

    def exists(self) -> bool:
        return self.client.collection_exists(self.config.collection)

    def ensure_collection(self, dimension: int, identity: dict[str, Any]) -> None:
        if not self.exists():
            self.client.create_collection(
                self.config.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                metadata={"library_ingestor": identity},
            )
        info = self.client.get_collection(self.config.collection)
        vector = info.config.params.vectors
        if not isinstance(vector, models.VectorParams) or vector.size != dimension:
            raise ConfigurationError("Qdrant vector dimension/schema differs from this model")
        if vector.distance != models.Distance.COSINE:
            raise ConfigurationError("Qdrant collection must use Cosine distance")
        if (info.config.metadata or {}).get("library_ingestor") != identity:
            raise ConfigurationError("Collection belongs to another index configuration/state")
        for field in ("document_id", "file_hash", "author", "language", "file_type", "title"):
            if field not in info.payload_schema:
                self.client.create_payload_index(
                    self.config.collection, field, models.PayloadSchemaType.KEYWORD, wait=True
                )

    def records(self, hashes: list[str], vectors: bool = False) -> Iterator[models.Record]:
        offset: int | str | None = None
        while True:
            records, next_offset = self.client.scroll(
                self.config.collection,
                scroll_filter=document_filter(hashes),
                limit=self.config.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=vectors,
            )
            yield from records
            if next_offset is None:
                return
            offset = cast(int | str, next_offset)

    def count(self, digest: str | None = None) -> int:
        if not self.exists():
            return 0
        return self.client.count(
            self.config.collection,
            exact=True,
            count_filter=document_filter([digest]) if digest else None,
        ).count

    def backup(self, hashes: list[str], path: Path) -> None:
        with path.open("w") as out:
            for record in self.records(hashes, vectors=True):
                point = models.PointStruct(
                    id=record.id, vector=cast(list[float], record.vector), payload=record.payload
                )
                out.write(point.model_dump_json() + "\n")
            out.flush()
            os.fsync(out.fileno())

    def delete(self, hashes: list[str]) -> None:
        if hashes:
            self.client.delete(
                self.config.collection,
                models.FilterSelector(filter=document_filter(hashes)),
                wait=True,
                ordering=models.WriteOrdering.STRONG,
            )

    def upload(self, path: Path) -> None:
        for batch in batched(read_points(path), self.config.batch_size):
            self.client.upsert(
                self.config.collection,
                points=list(batch),
                wait=True,
                ordering=models.WriteOrdering.STRONG,
            )

    def search(self, vector: list[float], limit: int) -> list[models.ScoredPoint]:
        return self.client.query_points(
            self.config.collection, query=vector, limit=limit, with_payload=True
        ).points

    def identity(self) -> dict[str, Any] | None:
        if not self.exists():
            return None
        value = (self.client.get_collection(self.config.collection).config.metadata or {}).get(
            "library_ingestor"
        )
        return cast(dict[str, Any] | None, value)
