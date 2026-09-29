import math

import pytest
from test_chunking import metadata

from library_ingestor.errors import StoreError
from library_ingestor.models import DocumentChunk, StructuralMetadata
from library_ingestor.utils.hashing import chunk_id
from library_ingestor.vectorstore.qdrant_store import make_point


def test_point_generation() -> None:
    chunk = DocumentChunk(
        id=chunk_id("hash", 0),
        document_id="hash",
        text="Content",
        chunk_index=0,
        metadata=StructuralMetadata(heading="Intro", page_number=2),
    )
    point = make_point(chunk, metadata(), [0.1, 0.2], "BAAI/bge-m3", "2026-09-20")
    assert point.payload is not None
    assert point.id == chunk.id
    assert point.payload["text"] == "Content"
    assert point.payload["heading"] == "Intro"
    assert point.payload["author"] is None
    assert point.payload["file_hash"] == "hash"
    assert point.payload["page_number"] == 2
    with pytest.raises(StoreError):
        make_point(chunk, metadata(), [math.nan], "model", "date")
