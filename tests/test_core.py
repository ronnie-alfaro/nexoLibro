import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from library_ingestor.config import ChunkingConfig, load_settings
from library_ingestor.discovery import discover
from library_ingestor.errors import ConfigurationError, ParsingError
from library_ingestor.metadata import base_metadata, epub_metadata
from library_ingestor.parsers.txt_parser import TxtParser
from library_ingestor.utils.hashing import chunk_id, file_sha256
from library_ingestor.utils.normalization import normalize


def test_discovery(tmp_path: Path) -> None:
    for name in ["one.txt", "book.PDF", ".hidden.txt", "~temp.txt", "odd.mobi", "empty.epub"]:
        (tmp_path / name).write_text("x" if name != "empty.epub" else "")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "two.epub").write_text("x")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.txt").write_text("secret")
    (tmp_path / "alias.txt").symlink_to(tmp_path / "one.txt")
    assert {p.name for p in discover(tmp_path, [".pdf", ".epub", ".txt"])} == {
        "one.txt",
        "book.PDF",
        "two.epub",
    }
    assert len(list(discover(tmp_path, [".txt", ".epub", ".pdf"], False))) == 2


def test_hash_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == hashlib.sha256(b"abc").hexdigest()
    assert chunk_id(file_sha256(path), 0) == chunk_id(file_sha256(path), 0)
    assert chunk_id("abc", 0) != chunk_id("abc", 1)
    assert chunk_id("abc", 0) != chunk_id("xyz", 0)


def test_txt_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "book.txt"
    path.write_text("\ufeff  First   paragraph.\n\nSecond paragraph.\n")
    meta = base_metadata(path, tmp_path, file_sha256(path))
    document = TxtParser().parse(path, meta)
    assert [b.text for b in document.blocks] == ["First paragraph.", "Second paragraph."]
    assert meta.title is None and meta.author is None and meta.language is None
    assert meta.relative_path == "book.txt"
    assert meta.file_size == path.stat().st_size
    path.write_bytes(b"\xff\xff")
    with pytest.raises(ParsingError):
        TxtParser().parse(path, meta)


def test_normalization() -> None:
    assert normalize(" e\u0301\t x\x00\n\n\n  y\u00ad ") == "é x\n\ny"


def test_epub_metadata(tmp_path: Path) -> None:
    path = tmp_path / "book.epub"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="p.opf"/></rootfiles></container>',
        )
        archive.writestr(
            "p.opf",
            """<package xmlns:dc="http://purl.org/dc/elements/1.1/">
        <metadata><dc:title>Ficciones</dc:title><dc:creator>Jorge Luis Borges</dc:creator>
        <dc:language>es</dc:language><dc:identifier>urn:isbn:123</dc:identifier></metadata>
        </package>""",
        )
    meta = base_metadata(path, tmp_path, file_sha256(path))
    epub_metadata(path, meta)
    assert meta.title == "Ficciones" and meta.author == "Jorge Luis Borges"
    assert meta.language == "es" and meta.isbn == "123" and meta.publisher is None


def test_config_env_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("embedding:\n  batch_size: 2\nqdrant:\n  collection: books\n")
    monkeypatch.setenv("LIBRARY_EMBEDDING__BATCH_SIZE", "7")
    settings = load_settings(config)
    assert settings.embedding.batch_size == 7
    assert settings.qdrant.collection == "books"
    assert settings.ingestion.state_dir == tmp_path / ".library_ingestor"
    with pytest.raises(ValueError):
        ChunkingConfig(target_tokens=20, max_tokens=10)
    monkeypatch.setenv("LIBRARY_QDRANT__HOST", "cloud.example.com")
    with pytest.raises(ConfigurationError):
        load_settings(config)


def test_environment_validates_after_yaml_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("chunking:\n  max_tokens: 1500\n")
    monkeypatch.setenv("LIBRARY_CHUNKING__TARGET_TOKENS", "1200")
    assert load_settings(config).chunking.target_tokens == 1200


def test_broken_epub_spine_rejected(tmp_path: Path) -> None:
    from library_ingestor.config import ParsingConfig
    from library_ingestor.parsers.docling_parser import DoclingParser

    path = tmp_path / "broken.epub"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml", '<container><rootfile full-path="p.opf"/></container>'
        )
        archive.writestr(
            "p.opf",
            '<package><manifest><item id="c" href="missing.xhtml"/></manifest>'
            '<spine><itemref idref="c"/></spine></package>',
        )
    with pytest.raises(ParsingError, match="missing.xhtml"):
        DoclingParser(ParsingConfig()).parse(path, base_metadata(path, tmp_path, file_sha256(path)))
