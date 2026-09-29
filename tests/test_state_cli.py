from pathlib import Path

import pytest
from typer.testing import CliRunner

from library_ingestor.cli import app
from library_ingestor.ingestion.state import StateDB
from library_ingestor.metadata import base_metadata
from library_ingestor.utils.hashing import file_sha256


def test_database_reopen_and_interrupted_state(tmp_path: Path) -> None:
    path = tmp_path / "book.txt"
    path.write_text("text")
    meta = base_metadata(path, tmp_path, file_sha256(path))
    db_path = tmp_path / "state.db"
    state = StateDB(db_path)
    state.start(meta, path.stat().st_mtime, "model")
    state.close()
    state = StateDB(db_path)
    try:
        assert (document := state.document(meta.file_hash)) is not None
        assert document["status"] == "processing"
        state.interrupted()
        assert (document := state.document(meta.file_hash)) is not None
        assert document["status"] == "failed"
        state.add_journal("a", {"hashes": [meta.file_hash]})
        state.complete(meta, path.stat().st_mtime, 2, "model", "a", None)
        assert state.journals()[0]["committed"] == 1
        state.alias(meta.file_hash, str(tmp_path / "copy.txt"))
        assert state.counts() == {"completed": 1}
        assert state.retired_hash(str(path), "changed") is None
        state.fail(meta.file_hash, str(path), "Failed force preserves original")
        assert (document := state.document(meta.file_hash)) is not None
        assert document["status"] == "completed"
    finally:
        state.close()


def test_dry_run_has_no_state_or_model_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "a.txt"
    path.write_text("Document")
    result = CliRunner().invoke(app, ["ingest", str(path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "INGEST:" in result.output and "Dry run: 1" in result.output
    assert not (tmp_path / ".library_ingestor").exists()


def test_cli_failure_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["ingest", str(tmp_path / "missing")])
    assert result.exit_code == 2
