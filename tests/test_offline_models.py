"""End-to-end check: block external sockets, use real models and Qdrant."""

import ipaddress
import socket
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from qdrant_client import models

from library_ingestor.cli import registry
from library_ingestor.config import IngestionConfig, ParsingConfig, QdrantConfig, Settings
from library_ingestor.embeddings.embedder import LocalEmbedder
from library_ingestor.ingestion.pipeline import Pipeline
from library_ingestor.ingestion.state import StateDB
from library_ingestor.vectorstore.qdrant_store import QdrantStore


@pytest.mark.models
@pytest.mark.integration
def test_actual_models_without_external_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.create_demo_library import create_library

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    original_connect = socket.socket.connect

    def local_connect(sock: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple):
            host = address[0]
            if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
                raise AssertionError(f"External connection attempted: {host}")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    settings = Settings(
        qdrant=QdrantConfig(collection="offline_test_" + uuid4().hex),
        ingestion=IngestionConfig(state_dir=tmp_path / "state"),
        parsing=ParsingConfig(artifacts_path=Path("models/docling").resolve()),
    )
    create_library(tmp_path / "books")
    store = QdrantStore(settings.qdrant)
    state = StateDB(settings.ingestion.state_dir / "state.db")
    embedder = LocalEmbedder(settings.embedding)
    try:
        pipeline = Pipeline(settings, registry(settings), embedder, store, state, lambda _: None)
        result = pipeline.ingest(tmp_path / "books")
        assert result.processed == 3 and result.failed == 0, result.failures
        assert store.count() == result.chunks_created
        assert pipeline.ingest(tmp_path / "books").skipped == 3
        hits = store.search(embedder.encode(["water cycles"])[0], 3)
        assert (hits[0].payload or {})["title"] == "Water cycles"
        assert embedder.dimension == 1024
        vectors = store.client.get_collection(settings.qdrant.collection).config.params.vectors
        assert isinstance(vectors, models.VectorParams)
        assert vectors.size == 1024
    finally:
        if store.exists():
            store.client.delete_collection(settings.qdrant.collection)
        store.close()
        state.close()


@pytest.mark.models
def test_scanned_pdf_local_ocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    from library_ingestor.errors import ParsingError
    from library_ingestor.metadata import base_metadata
    from library_ingestor.parsers.docling_parser import DoclingParser
    from library_ingestor.utils.hashing import file_sha256

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    original_connect = socket.socket.connect

    def local_connect(sock: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple) and not ipaddress.ip_address(address[0]).is_loopback:
            raise AssertionError("OCR attempted external network access")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    image = Image.new("RGB", (1400, 400), "white")
    ImageDraw.Draw(image).text(
        (40, 100),
        "Water and soil in a scanned document.",
        font=ImageFont.load_default(size=48),
        fill="black",
    )
    path = tmp_path / "scanned.pdf"
    canvas = Canvas(str(path), pagesize=(700, 200))
    canvas.drawImage(ImageReader(image), 0, 0, width=700, height=200)
    canvas.save()
    metadata = base_metadata(path, tmp_path, file_sha256(path))
    artifacts = Path("models/docling").resolve()
    with pytest.raises(ParsingError, match="selectable"):
        DoclingParser(ParsingConfig(artifacts_path=artifacts)).parse(path, metadata)
    document = DoclingParser(ParsingConfig(ocr="auto", artifacts_path=artifacts)).parse(
        path, metadata
    )
    assert "water" in " ".join(b.text for b in document.blocks).lower()
