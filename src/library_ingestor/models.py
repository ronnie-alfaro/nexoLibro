"""Typed contracts shared by all pipeline components."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

Status = Literal["pending", "processing", "completed", "failed", "skipped"]


class DocumentMetadata(BaseModel):
    document_id: str
    file_hash: str
    filename: str
    filepath: str
    relative_path: str
    file_type: str
    file_size: int
    title: str | None = None
    author: str | None = None
    language: str | None = None
    publisher: str | None = None
    publication_date: str | None = None
    isbn: str | None = None


class StructuralMetadata(BaseModel):
    chapter: str | None = None
    section: str | None = None
    heading: str | None = None
    page_number: int | None = None


class TextBlock(BaseModel):
    text: str
    kind: Literal["heading", "paragraph", "table", "list"] = "paragraph"
    metadata: StructuralMetadata = StructuralMetadata()


class ParsedDocument(BaseModel):
    metadata: DocumentMetadata
    blocks: list[TextBlock]


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: StructuralMetadata


@dataclass
class IngestionResult:
    discovered: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    chunks_created: int = 0
    elapsed_seconds: float = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
