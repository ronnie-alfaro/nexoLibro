"""Real HTTP Qdrant, isolated disposable collections. No model download required."""

from pathlib import Path
from uuid import uuid4

import pytest
from conftest import FakeEmbedder

from library_ingestor.cli import registry
from library_ingestor.config import ChunkingConfig, IngestionConfig, QdrantConfig, Settings
from library_ingestor.ingestion.pipeline import Pipeline
from library_ingestor.ingestion.state import StateDB
from library_ingestor.vectorstore.qdrant_store import QdrantStore


@pytest.mark.integration
def test_real_qdrant_replacement_and_indexes(tmp_path: Path) -> None:
    settings = Settings(
        qdrant=QdrantConfig(collection="test_" + uuid4().hex),
        ingestion=IngestionConfig(state_dir=tmp_path / "state"),
        chunking=ChunkingConfig(target_tokens=70, max_tokens=100, overlap_tokens=10),
    )
    store = QdrantStore(settings.qdrant)
    state = StateDB(settings.ingestion.state_dir / "state.db")
    try:
        pipeline = Pipeline(
            settings, registry(settings), FakeEmbedder(), store, state, lambda _: None
        )
        path = tmp_path / "book.txt"
        path.write_text("Original contents. " * 10)
        result = pipeline.ingest(path)
        assert result.failed == 0 and store.count() == result.chunks_created
        assert pipeline.ingest(path).skipped == 1
        schema = store.client.get_collection(settings.qdrant.collection).payload_schema
        assert set(schema) == {
            "document_id",
            "file_hash",
            "author",
            "language",
            "file_type",
            "title",
        }
        assert (store.search([1.0, 0.03, 0.25], 1)[0].payload or {})["filename"] == "book.txt"
        path.write_text("Replacement")
        assert pipeline.ingest(path).processed == 1
        assert store.count() == 1
    finally:
        if store.exists():
            store.client.delete_collection(settings.qdrant.collection)
        store.close()
        state.close()
