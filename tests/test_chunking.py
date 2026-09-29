from conftest import CharacterTokenizer

from library_ingestor.chunking.chunker import StructuralChunker
from library_ingestor.config import ChunkingConfig
from library_ingestor.models import DocumentMetadata, ParsedDocument, StructuralMetadata, TextBlock


def metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="hash",
        file_hash="hash",
        filename="a.txt",
        filepath="/a.txt",
        relative_path="a.txt",
        file_type="txt",
        file_size=1,
    )


def test_structural_chunks_and_limits() -> None:
    first = StructuralMetadata(section="Origins", heading="Origins", page_number=1)
    second = StructuralMetadata(section="Destiny", heading="Destiny", page_number=2)
    doc = ParsedDocument(
        metadata=metadata(),
        blocks=[
            TextBlock(text="Alpha " * 40, metadata=first),
            TextBlock(text="Omega " * 40, metadata=second),
        ],
    )
    chunker = StructuralChunker(
        ChunkingConfig(target_tokens=70, max_tokens=100, overlap_tokens=10), CharacterTokenizer()
    )
    chunks = list(chunker.chunks(doc))
    assert len(chunks) >= 4
    assert all(chunker.count(c.text) <= 100 for c in chunks)
    assert [c.id for c in chunks] == [c.id for c in chunker.chunks(doc)]
    assert all(c.document_id == "hash" for c in chunks)
    assert chunks[0].metadata.page_number == 1
    assert chunks[-1].metadata.section == "Destiny"
    assert not any("Alpha" in c.text and "Omega" in c.text for c in chunks)


def test_unicode_text_never_corrupted_or_lost() -> None:
    text = "漢字🚀á" * 100
    doc = ParsedDocument(metadata=metadata(), blocks=[TextBlock(text=text)])
    chunker = StructuralChunker(
        ChunkingConfig(target_tokens=25, max_tokens=40, overlap_tokens=0), CharacterTokenizer()
    )
    chunks = list(chunker.chunks(doc))
    assert "".join(c.text for c in chunks) == text
    assert all(len(c.text) <= 40 for c in chunks)


def test_short_paragraphs_pack_together() -> None:
    doc = ParsedDocument(metadata=metadata(), blocks=[TextBlock(text="short") for _ in range(5)])
    chunker = StructuralChunker(
        ChunkingConfig(target_tokens=70, max_tokens=100, overlap_tokens=10), CharacterTokenizer()
    )
    assert len(list(chunker.chunks(doc))) == 1
