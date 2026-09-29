from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from library_ingestor.config import ChunkingConfig, IngestionConfig, Settings
from library_ingestor.ingestion.pipeline import Pipeline
from library_ingestor.ingestion.state import StateDB
from library_ingestor.parsers.base import ParserRegistry
from library_ingestor.parsers.txt_parser import TxtParser
from library_ingestor.vectorstore.qdrant_store import QdrantStore


class CharacterTokenizer:
    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "".join(chr(i) for i in ids)


class FakeEmbedder:
    tokenizer = CharacterTokenizer()
    dimension = 3
    max_tokens = 8192

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text)) / 1000, 0.25] for text in texts]


type PipelineFixture = tuple[Pipeline, StateDB, QdrantStore]


@pytest.fixture
def setup_pipeline(tmp_path: Path) -> Iterator[PipelineFixture]:
    settings = Settings(
        ingestion=IngestionConfig(state_dir=tmp_path / "state"),
        chunking=ChunkingConfig(target_tokens=70, max_tokens=100, overlap_tokens=10),
    )
    state = StateDB(settings.ingestion.state_dir / "state.db")
    client = QdrantClient(":memory:")
    store = QdrantStore(settings.qdrant, client)
    parsers = ParserRegistry()
    parsers.register(".txt", TxtParser())
    pipeline = Pipeline(settings, parsers, FakeEmbedder(), store, state, lambda _: None)
    yield pipeline, state, store
    store.close()
    state.close()
