from pathlib import Path

import pytest
from conftest import PipelineFixture

from library_ingestor.errors import ConfigurationError
from library_ingestor.utils.hashing import file_sha256


def test_dedup_reindex_force_and_state(tmp_path: Path, setup_pipeline: PipelineFixture) -> None:
    pipeline, state, store = setup_pipeline
    root = tmp_path / "books"
    root.mkdir()
    first = root / "one.txt"
    first.write_text("Library documents about water cycles. " * 10)
    original = file_sha256(first)
    result = pipeline.ingest(root)
    assert result.processed == 1 and result.failed == 0
    count = store.count()
    assert count == result.chunks_created
    assert (document := state.document(original)) is not None
    assert document["status"] == "completed"
    assert pipeline.ingest(root).skipped == 1
    assert store.count() == count
    ids = {r.id for r in store.records([original])}
    assert pipeline.ingest(root, force=True).processed == 1
    assert {r.id for r in store.records([original])} == ids
    first.write_text("Changed and much shorter.")
    assert pipeline.ingest(root).processed == 1
    assert store.count(original) == 0
    assert state.document(original) is None
    assert store.count() == 1


def test_alias_copy_protects_old_content(tmp_path: Path, setup_pipeline: PipelineFixture) -> None:
    pipeline, state, store = setup_pipeline
    root = tmp_path / "books"
    root.mkdir()
    first, second = root / "a.txt", root / "b.txt"
    first.write_text("Same original contents")
    second.write_text(first.read_text())
    digest = file_sha256(first)
    result = pipeline.ingest(root)
    assert (result.processed, result.skipped) == (1, 1)
    first.write_text("Changed contents")
    result = pipeline.ingest(root)
    assert result.processed == 1 and result.skipped == 1
    assert store.count(digest) == 1
    assert store.count() == 2
    assert state.path_hash(str(second)) == digest


def test_failed_document_does_not_stop_library(
    tmp_path: Path, setup_pipeline: PipelineFixture
) -> None:
    pipeline, state, store = setup_pipeline
    root = tmp_path / "books"
    root.mkdir()
    (root / "bad.txt").write_bytes(b"\xff")
    (root / "good.txt").write_text("Good text")
    result = pipeline.ingest(root)
    assert result.failed == 1 and result.processed == 1
    assert store.count() == 1
    assert state.counts() == {"failed": 1, "completed": 1}


def test_partial_qdrant_failure_rolls_back(
    tmp_path: Path, setup_pipeline: PipelineFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, state, store = setup_pipeline
    path = tmp_path / "book.txt"
    path.write_text("Previous version. " * 10)
    pipeline.ingest(path)
    digest = file_sha256(path)
    old_ids = {r.id for r in store.records([digest])}
    upload = store.upload

    def fail_new(spool: Path) -> None:
        if spool.name == "new.jsonl":
            from library_ingestor.vectorstore.qdrant_store import read_points

            store.client.upsert(store.config.collection, points=[next(read_points(spool))])
            raise RuntimeError("simulated partial batch failure")
        upload(spool)

    monkeypatch.setattr(store, "upload", fail_new)
    path.write_text("New different version. " * 15)
    result = pipeline.ingest(path)
    assert result.failed == 1
    assert store.count(file_sha256(path)) == 0
    assert {r.id for r in store.records([digest])} == old_ids
    assert (document := state.document(digest)) is not None
    assert document["status"] == "completed"
    assert not state.journals()


def test_crash_journal_restores_previous_points(
    tmp_path: Path, setup_pipeline: PipelineFixture
) -> None:
    pipeline, state, store = setup_pipeline
    path = tmp_path / "book.txt"
    path.write_text("Stable original")
    pipeline.ingest(path)
    digest = file_sha256(path)
    directory = pipeline.settings.ingestion.state_dir / "spool" / "interrupted"
    directory.mkdir(parents=True)
    store.backup([digest], directory / "old.jsonl")
    state.add_journal("interrupted", {"directory": str(directory), "hashes": [digest]})
    store.delete([digest])
    assert store.count() == 0
    pipeline.recover()
    assert store.count() == 1
    assert (document := state.document(digest)) is not None
    assert document["status"] == "completed"
    assert not state.journals()


def test_model_change_rejected(tmp_path: Path, setup_pipeline: PipelineFixture) -> None:
    pipeline, _, _ = setup_pipeline
    path = tmp_path / "book.txt"
    path.write_text("Original")
    pipeline.ingest(path)
    pipeline.settings.embedding.model = "different-model"
    with pytest.raises(ConfigurationError):
        pipeline.ingest(path)


def test_failed_force_preserves_completed_version(
    tmp_path: Path,
    setup_pipeline: PipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, state, store = setup_pipeline
    path = tmp_path / "book.txt"
    path.write_text("Stable version")
    pipeline.ingest(path)
    digest = file_sha256(path)
    original = list(store.records([digest]))

    def fail_embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("No memory")

    monkeypatch.setattr(pipeline.embedder, "encode", fail_embed)
    result = pipeline.ingest(path, force=True)
    assert result.failed == 1
    assert list(store.records([digest])) == original
    assert (document := state.document(digest)) is not None
    assert document["status"] == "completed"
    assert not state.journals()


def test_incomplete_rollback_blocks_then_recovers(
    tmp_path: Path,
    setup_pipeline: PipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from library_ingestor.errors import RecoveryError

    pipeline, state, store = setup_pipeline
    path = tmp_path / "book.txt"
    path.write_text("Stable version")
    pipeline.ingest(path)
    digest = file_sha256(path)
    original_upload = store.upload

    def fail_upload(spool: Path) -> None:
        raise RuntimeError("Server unavailable")

    monkeypatch.setattr(store, "upload", fail_upload)
    path.write_text("Changed version")
    with pytest.raises(RecoveryError):
        pipeline.ingest(path)
    assert len(state.journals()) == 1
    monkeypatch.setattr(store, "upload", original_upload)
    pipeline.recover()
    assert store.count(digest) == 1 and not state.journals()


def test_shorter_force_removes_surplus_points(
    tmp_path: Path,
    setup_pipeline: PipelineFixture,
) -> None:
    # A stale extra point for the same document must not survive a repair.
    from qdrant_client import models

    from library_ingestor.utils.hashing import chunk_id

    pipeline, _, store = setup_pipeline
    path = tmp_path / "book.txt"
    path.write_text("One small chunk")
    pipeline.ingest(path)
    digest = file_sha256(path)
    store.client.upsert(
        store.config.collection,
        points=[
            models.PointStruct(
                id=chunk_id(digest, 99),
                vector=[1.0, 0.0, 0.0],
                payload={"file_hash": digest},
            )
        ],
    )
    assert store.count() == 2
    assert pipeline.ingest(path).processed == 1
    assert store.count() == 1
