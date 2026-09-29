"""TXT has paragraphs, but no reliable bibliographic or chapter metadata."""

import re
from pathlib import Path

from library_ingestor.errors import ParsingError
from library_ingestor.models import DocumentMetadata, ParsedDocument, TextBlock
from library_ingestor.utils.normalization import normalize


class TxtParser:
    def __init__(self, encoding: str = "utf-8-sig") -> None:
        self.encoding = encoding

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        try:
            text = normalize(path.read_text(encoding=self.encoding))
        except (UnicodeError, OSError) as exc:
            raise ParsingError(f"Cannot decode TXT as {self.encoding}: {path}") from exc
        if not text:
            raise ParsingError("Document contains no usable text")
        return ParsedDocument(
            metadata=metadata,
            blocks=[TextBlock(text=p) for p in re.split(r"\n\s*\n", text) if p.strip()],
        )
