"""Parser extension boundary: register another suffix without changing ingestion."""

from pathlib import Path
from typing import Protocol

from library_ingestor.errors import ParsingError
from library_ingestor.models import DocumentMetadata, ParsedDocument


class Parser(Protocol):
    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument: ...


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}

    def register(self, extension: str, parser: Parser) -> None:
        self._parsers[extension.lower()] = parser

    def get(self, path: Path) -> Parser:
        try:
            return self._parsers[path.suffix.lower()]
        except KeyError as exc:
            raise ParsingError(f"No parser registered for {path.suffix}") from exc
