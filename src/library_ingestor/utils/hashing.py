"""Content identity independent of filename or location."""

import hashlib
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def chunk_id(document_hash: str, index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"library-ingestor:{document_hash}:{index}"))
