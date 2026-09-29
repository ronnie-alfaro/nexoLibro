"""Generate small original test documents, never copyrighted book extracts."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from reportlab.pdfgen.canvas import Canvas


def create_library(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "gardening.txt").write_text(
        "Vegetable gardening\n\n"
        "Tomatoes grow in sunny gardens. Water the roots and enrich the soil with compost. "
        "Seedlings need regular watering and protection from frost.\n\n"
        "Harvest ripe tomatoes in summer. Healthy soil supports roots and earthworms.\n",
        encoding="utf-8",
    )
    pdf = Canvas(str(root / "astronomy.pdf"))
    pdf.setTitle("A guide to distant stars")
    pdf.setAuthor("Demo Observatory")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(60, 760, "Distant stars and galaxies")
    pdf.setFont("Helvetica", 11)
    for index, line in enumerate(
        [
            "Telescopes gather light from distant stars and galaxies.",
            "Astronomers measure stellar spectra to understand chemical composition.",
            "A galaxy contains billions of stars, gas clouds and planets.",
        ]
    ):
        pdf.drawString(60, 710 - index * 20, line)
    pdf.save()
    with ZipFile(root / "water_cycles.epub", "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
        <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
        <rootfiles><rootfile full-path="OEBPS/content.opf"
        media-type="application/oebps-package+xml"/></rootfiles></container>""",
        )
        archive.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0"?>
        <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
        <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="id">demo-water-cycles-2026</dc:identifier>
        <dc:title>Water cycles</dc:title><dc:creator>Demo Author</dc:creator>
        <dc:language>en</dc:language><dc:publisher>Local Demo Press</dc:publisher>
        <dc:date>2026-09-20</dc:date></metadata>
        <manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
        <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
        </manifest><spine><itemref idref="chapter"/></spine></package>""",
        )
        archive.writestr(
            "OEBPS/chapter.xhtml",
            """<html xmlns="http://www.w3.org/1999/xhtml">
        <head><title>Water cycles</title></head><body><h1>Water in motion</h1>
        <p>Water evaporates from lakes and oceans, condenses into clouds, and returns
        to the ground as precipitation.</p>
        <h2>Watersheds</h2><p>Soil, plants and rivers guide water through a watershed.
        Careful land use helps keep waterways healthy.</p>
        </body></html>""",
        )
        archive.writestr(
            "OEBPS/nav.xhtml",
            """<html xmlns="http://www.w3.org/1999/xhtml"
        xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Contents</title></head>
        <body><nav epub:type="toc"><ol><li><a href="chapter.xhtml">Water in motion</a>
        </li></ol></nav></body></html>""",
        )


if __name__ == "__main__":
    create_library(Path(".library_ingestor/demo/books"))
