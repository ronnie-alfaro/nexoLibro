"""Regressions for image covers and the five-page text probe, without loading models."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from PIL import Image
from pypdf import PageObject
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from library_ingestor.config import ParsingConfig
from library_ingestor.errors import ParsingError
from library_ingestor.metadata import base_metadata, pdf_metadata
from library_ingestor.parsers.docling_parser import DoclingParser
from library_ingestor.utils.hashing import file_sha256


def create_pdf(path: Path, pages: int, text_page: int | None) -> None:
    canvas = Canvas(str(path))
    canvas.setTitle("Five-page detection test")
    cover = ImageReader(Image.new("RGB", (20, 20), "gray"))
    for number in range(1, pages + 1):
        if number == text_page:
            canvas.drawString(60, 700, f"Selectable text on page {number}.")
        else:
            canvas.drawImage(cover, 60, 600, width=100, height=100)
        canvas.showPage()
    canvas.save()


@pytest.mark.parametrize("text_page", [1, 2, 5, 6, None])
def test_probes_only_first_five_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text_page: int | None
) -> None:
    path = tmp_path / "book.pdf"
    create_pdf(path, pages=7, text_page=text_page)
    original_extract = PageObject.extract_text
    calls = 0

    def counted_extract(page: PageObject, *args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        assert calls <= 5, "The text probe must never extract the sixth page"
        return original_extract(page, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", counted_extract)
    metadata = base_metadata(path, tmp_path, file_sha256(path))
    availability = pdf_metadata(path, metadata)
    assert availability == [number == text_page for number in range(1, 6)]
    assert calls == 5
    assert metadata.title == "Five-page detection test"


@pytest.mark.parametrize("pages,text_page,expected", [(1, 1, [True]), (2, 2, [False, True])])
def test_short_pdf(tmp_path: Path, pages: int, text_page: int, expected: list[bool]) -> None:
    path = tmp_path / "short.pdf"
    create_pdf(path, pages, text_page)
    assert pdf_metadata(path, base_metadata(path, tmp_path, file_sha256(path))) == expected


@pytest.mark.parametrize("text_page", [1, 2, 5])
@pytest.mark.parametrize("ocr_policy", ["never", "auto"])
def test_any_sampled_text_allows_full_conversion_without_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text_page: int,
    ocr_policy: Literal["never", "auto"],
) -> None:
    path = tmp_path / "book.pdf"
    create_pdf(path, pages=7, text_page=text_page)
    parser = DoclingParser(ParsingConfig(ocr=ocr_policy))
    decisions: list[bool] = []
    converted: list[Path] = []

    def converter(ocr: bool) -> Any:
        decisions.append(ocr)

        def convert(source: Path) -> Any:
            converted.append(source)
            item = SimpleNamespace(label=SimpleNamespace(value="text"), text="Book text", prov=[])
            document = SimpleNamespace(iterate_items=lambda: iter([(item, 0)]))
            return SimpleNamespace(status=SimpleNamespace(value="success"), document=document)

        return SimpleNamespace(convert=convert)

    monkeypatch.setattr(parser, "_converter", converter)
    document = parser.parse(path, base_metadata(path, tmp_path, file_sha256(path)))
    assert decisions == [False]
    assert converted == [path]
    assert document.blocks[0].text == "Book text"


@pytest.mark.parametrize("pages,text_page", [(1, None), (7, None), (7, 6)])
def test_no_sampled_text_rejected_without_ocr(
    tmp_path: Path, pages: int, text_page: int | None
) -> None:
    path = tmp_path / "scanned.pdf"
    create_pdf(path, pages, text_page)
    with pytest.raises(ParsingError, match=f"first {min(pages, 5)} page"):
        DoclingParser(ParsingConfig()).parse(path, base_metadata(path, tmp_path, file_sha256(path)))
