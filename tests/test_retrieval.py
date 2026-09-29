import json
from pathlib import Path

import pytest
from conftest import PipelineFixture
from qdrant_client import models

from library_ingestor.config import LlamaConfig, load_settings
from library_ingestor.retrieval.catalog import Catalog
from library_ingestor.retrieval.retriever import Retriever
from library_ingestor.utils.hashing import file_sha256


def test_new_config_does_not_invalidate_ingestion(
    tmp_path: Path, setup_pipeline: PipelineFixture
) -> None:
    pipeline, _, _ = setup_pipeline
    path = tmp_path / "book.txt"
    path.write_text("A private book")
    pipeline.ingest(path)
    pipeline.settings.llama.port = 9090
    pipeline.settings.retrieval.source_limit = 3
    assert pipeline.ingest(path).skipped == 1


def test_catalog_pagination_and_journal_exclusion(
    tmp_path: Path, setup_pipeline: PipelineFixture
) -> None:
    pipeline, state, _ = setup_pipeline
    first = tmp_path / "a.txt"
    first.write_text("An original book")
    pipeline.ingest(first)
    second = tmp_path / "100%.txt"
    second.write_text("Another original book")
    pipeline.ingest(second)
    catalog = Catalog(pipeline.settings.ingestion.state_dir)
    assert catalog.stats()["books"] == 2
    page = catalog.books("", 0, 1)
    assert page["total"] == 2 and len(page["items"]) == 1
    assert catalog.books("%", 0, 5)["items"][0]["title"] == "100%.txt"
    assert catalog.books("' OR 1=1 --", 0, 5)["total"] == 0
    digest = file_sha256(first)
    state.add_journal("pending", {"hashes": [digest]})
    assert digest not in catalog.readable_hashes([digest])
    state.remove_journal("pending")
    assert digest in catalog.readable_hashes([digest])


def test_retrieval_reuses_embeddings_but_not_results(
    tmp_path: Path, setup_pipeline: PipelineFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _, store = setup_pipeline
    path = tmp_path / "book.txt"
    path.write_text("A text about watersheds")
    pipeline.ingest(path)
    retriever = Retriever(
        pipeline.settings, pipeline.embedder, store, Catalog(pipeline.settings.ingestion.state_dir)
    )
    calls = 0
    original = pipeline.embedder.encode

    def counted(texts: list[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        return original(texts)

    monkeypatch.setattr(pipeline.embedder, "encode", counted)
    assert retriever.retrieve("watersheds")
    assert retriever.retrieve("watersheds")
    assert calls == 1
    store.delete([file_sha256(path)])
    assert retriever.retrieve("watersheds") == []
    assert calls == 1


def test_diversity_duplicates_and_document_scope(setup_pipeline: PipelineFixture) -> None:
    pipeline, _, store = setup_pipeline
    retriever = Retriever(
        pipeline.settings, pipeline.embedder, store, Catalog(pipeline.settings.ingestion.state_dir)
    )

    def hit(index: int, digest: str, text: str) -> models.ScoredPoint:
        return models.ScoredPoint(
            id=index,
            version=1,
            score=0.9 - index * 0.01,
            payload={
                "file_hash": digest,
                "text": text,
                "filename": "book.txt",
                "chunk_index": index,
            },
        )

    hits = [
        hit(1, "a", "water cycles and watersheds"),
        hit(2, "a", "water cycles and watersheds"),
        hit(3, "a", "rivers distant landscapes"),
        hit(4, "a", "planets stars moons"),
        hit(5, "b", "books libraries memory"),
        hit(6, "unfinished", "do not read"),
    ]
    results = retriever._select(hits, {"a", "b"}, None)
    assert [result.point_id for result in results] == ["1", "3", "5"]
    scoped = retriever._select(hits[:4], {"a"}, "a")
    assert len(scoped) == 3


def test_only_loopback_models_allowed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LlamaConfig(host="cloud.example.com")  # type: ignore[arg-type]
    path = tmp_path / "settings.yaml"
    path.write_text(json.dumps({"llama": {"port": 9999}}))
    assert load_settings(path).llama.port == 9999
