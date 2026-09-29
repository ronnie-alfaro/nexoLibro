"""Docling adapter preserving headings, reading order, tables and provenance."""

from pathlib import Path
from typing import Any, Literal

from library_ingestor.config import ParsingConfig
from library_ingestor.errors import ParsingError
from library_ingestor.metadata import epub_metadata, pdf_metadata, validate_epub_spine
from library_ingestor.models import (
    DocumentMetadata,
    ParsedDocument,
    StructuralMetadata,
    TextBlock,
)
from library_ingestor.utils.normalization import normalize


class DoclingParser:
    def __init__(self, config: ParsingConfig) -> None:
        self.config = config
        self._converters: dict[bool, Any] = {}

    def _converter(self, ocr: bool) -> Any:
        if ocr not in self._converters:
            from docling.datamodel.backend_options import EpubBackendOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            from docling.document_converter import (
                DocumentConverter,
                EpubFormatOption,
                PdfFormatOption,
            )

            options = PdfPipelineOptions(
                do_ocr=ocr,
                ocr_options=RapidOcrOptions(backend="onnxruntime"),
                enable_remote_services=False,
                do_picture_description=False,
                do_picture_classification=False,
                artifacts_path=self.config.artifacts_path,
            )
            self._converters[ocr] = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.EPUB],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                    InputFormat.EPUB: EpubFormatOption(
                        backend_options=EpubBackendOptions(
                            enable_remote_fetch=False,
                            enable_local_fetch=False,
                            fetch_images=False,
                        )
                    ),
                },
            )
        return self._converters[ocr]

    def parse(self, path: Path, metadata: DocumentMetadata) -> ParsedDocument:
        try:
            needs_ocr = False
            if path.suffix.lower() == ".epub":
                validate_epub_spine(path)
                epub_metadata(path, metadata)
            else:
                pages = pdf_metadata(path, metadata)
                needs_ocr = bool(pages) and not any(pages)
                if needs_ocr and self.config.ocr == "never":
                    raise ParsingError(
                        f"PDF has no extractable/selectable text in its first {len(pages)} "
                        "page(s) checked (possibly scanned or blank). "
                        "Review it and use parsing.ocr: auto to allow local OCR."
                    )
            result = self._converter(needs_ocr).convert(path)
            if str(result.status.value) != "success":
                raise ParsingError(f"Docling conversion incomplete: {result.status}")
            document = result.document
            blocks: list[TextBlock] = []
            headings: dict[int, str] = {}
            for item, _level in document.iterate_items():
                label = item.label.value
                if label in {"page_header", "page_footer", "picture"}:
                    continue
                text = (
                    item.export_to_markdown(doc=document)
                    if label == "table"
                    else getattr(item, "text", "")
                )
                text = normalize(text)
                if not text:
                    continue
                if label in {"section_header", "title"}:
                    if label == "title" and metadata.title is None:
                        metadata.title = text
                    level = int(getattr(item, "level", 1))
                    headings = {key: val for key, val in headings.items() if key < level}
                    headings[level] = text
                heading = next(reversed(headings.values()), None)
                context = StructuralMetadata(
                    heading=heading,
                    section=" / ".join(headings.values()) or None,
                    # Level-one headings are not necessarily chapters; keep chapter null.
                    page_number=item.prov[0].page_no if item.prov else None,
                )
                kind: Literal["heading", "paragraph", "table", "list"] = (
                    "heading"
                    if label in {"section_header", "title"}
                    else "table"
                    if label == "table"
                    else "list"
                    if label == "list_item"
                    else "paragraph"
                )
                blocks.append(TextBlock(text=text, kind=kind, metadata=context))
            if not blocks:
                raise ParsingError("Docling extracted no usable text")
            return ParsedDocument(metadata=metadata, blocks=blocks)
        except ParsingError:
            raise
        except Exception as exc:
            raise ParsingError(f"Docling could not parse {path.name}: {exc}") from exc
