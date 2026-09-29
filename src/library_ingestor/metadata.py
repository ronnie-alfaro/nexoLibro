"""Filesystem identity and embedded bibliographic metadata. No inferred values."""

import posixpath
from itertools import islice
from pathlib import Path
from typing import cast
from zipfile import BadZipFile, ZipFile

from defusedxml import ElementTree
from pypdf import PdfReader
from pypdf.generic import DictionaryObject

from library_ingestor.errors import ParsingError
from library_ingestor.models import DocumentMetadata


def validate_epub_spine(path: Path) -> None:
    """Docling can skip unreadable EPUB chapters; reject those before conversion."""
    with ZipFile(path) as archive:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile")
        if rootfile is None:
            raise ParsingError("EPUB has no package rootfile")
        package = rootfile.attrib["full-path"]
        document = ElementTree.fromstring(archive.read(package))
        manifest = {
            item.attrib["id"]: item.attrib["href"]
            for item in document.findall(".//{*}manifest/{*}item")
        }
        spine = document.findall(".//{*}spine/{*}itemref")
        if not spine:
            raise ParsingError("EPUB has an empty reading-order spine")
        from urllib.parse import unquote

        for reference in spine:
            href = unquote(manifest[reference.attrib["idref"]].split("#", 1)[0])
            chapter = posixpath.normpath(posixpath.join(posixpath.dirname(package), href))
            archive.read(chapter).decode("utf-8")


def base_metadata(path: Path, root: Path, digest: str) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=digest,
        file_hash=digest,
        filename=path.name,
        filepath=str(path.resolve()),
        relative_path=str(path.relative_to(root)),
        file_type=path.suffix.lower().lstrip("."),
        file_size=path.stat().st_size,
    )


def epub_metadata(path: Path, metadata: DocumentMetadata) -> None:
    try:
        with ZipFile(path) as archive:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            rootfile = container.find(".//{*}rootfile")
            if rootfile is None:
                raise ParsingError("EPUB has no package rootfile")
            package = posixpath.normpath(rootfile.attrib["full-path"])
            document = ElementTree.fromstring(archive.read(package))
            fields = {
                "title": "title",
                "creator": "author",
                "language": "language",
                "publisher": "publisher",
                "date": "publication_date",
            }
            for tag, field in fields.items():
                values = [
                    node.text.strip()
                    for node in document.findall(f".//{{*}}metadata/{{*}}{tag}")
                    if node.text and node.text.strip()
                ]
                if values:
                    setattr(metadata, field, "; ".join(values))
            for node in document.findall(".//{*}metadata/{*}identifier"):
                value = (node.text or "").strip()
                scheme = " ".join(node.attrib.values()).lower()
                if value.lower().startswith("urn:isbn:") or "isbn" in scheme:
                    metadata.isbn = value.removeprefix("urn:isbn:")
                    break
    except (BadZipFile, KeyError, ValueError, OSError) as exc:
        raise ParsingError(f"Invalid EPUB metadata: {exc}") from exc


def pdf_metadata(path: Path, metadata: DocumentMetadata) -> list[bool]:
    """Read metadata and probe extractable text in at most the first five pages."""
    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ParsingError("Encrypted PDF requires a password")
    info = reader.metadata
    if info:
        metadata.title = str(info.title).strip() if info.title else None
        metadata.author = str(info.author).strip() if info.author else None
        for key, field in (
            ("/Publisher", "publisher"),
            ("/ISBN", "isbn"),
            ("/PublicationDate", "publication_date"),
        ):
            value = info.get(key)
            if value:
                setattr(metadata, field, str(value).strip())
    xmp = reader.xmp_metadata
    if xmp:
        if not metadata.title and xmp.dc_title:
            metadata.title = xmp.dc_title.get("x-default") or next(iter(xmp.dc_title.values()))
        if not metadata.author and xmp.dc_creator:
            metadata.author = "; ".join(xmp.dc_creator)
        if not metadata.publisher and xmp.dc_publisher:
            metadata.publisher = "; ".join(xmp.dc_publisher)
        if xmp.dc_language:
            metadata.language = "; ".join(xmp.dc_language)
    # CreationDate is not publication_date; Producer is not publisher.
    language = cast(DictionaryObject, reader.trailer["/Root"]).get("/Lang")
    if language:
        metadata.language = str(language)
    return [bool((page.extract_text() or "").strip()) for page in islice(reader.pages, 5)]
