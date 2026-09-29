"""Structure-first packing with exact tokenizer budgets and bounded overlap."""

import re
from collections.abc import Iterator

from library_ingestor.config import ChunkingConfig
from library_ingestor.embeddings.embedder import Tokenizer
from library_ingestor.models import DocumentChunk, ParsedDocument, StructuralMetadata
from library_ingestor.utils.hashing import chunk_id
from library_ingestor.utils.normalization import normalize


class StructuralChunker:
    def __init__(self, config: ChunkingConfig, tokenizer: Tokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split(self, text: str, budget: int) -> Iterator[str]:
        """Prefer sentences. Oversize sentences split at exact character boundaries.

        Binary search avoids decoding token slices, which can corrupt multi-byte Unicode.
        """
        if self.count(text) <= budget:
            yield text
            return
        for sentence in re.split(r"(?<=[.!?。！？])\s+", text):
            while self.count(sentence) > budget:
                low, high = 1, len(sentence)
                while low < high:
                    mid = (low + high + 1) // 2
                    if self.count(sentence[:mid]) <= budget:
                        low = mid
                    else:
                        high = mid - 1
                cut = low
                space = sentence.rfind(" ", 0, cut + 1)
                if space > cut // 2:
                    cut = space
                yield sentence[:cut].strip()
                sentence = sentence[cut:].lstrip()
            if sentence:
                yield sentence

    def _tail(self, text: str) -> str:
        budget = self.config.overlap_tokens
        if not budget:
            return ""
        low, high = 0, len(text)
        while low < high:
            mid = (low + high) // 2
            if self.count(text[mid:]) <= budget:
                high = mid
            else:
                low = mid + 1
        return text[low:].strip()

    def chunks(self, document: ParsedDocument) -> Iterator[DocumentChunk]:
        index = 0
        text = ""
        context = StructuralMetadata()

        def make(value: str, meta: StructuralMetadata) -> DocumentChunk:
            return DocumentChunk(
                id=chunk_id(document.metadata.file_hash, index),
                document_id=document.metadata.document_id,
                text=value,
                chunk_index=index,
                metadata=meta.model_copy(),
            )

        for block in document.blocks:
            value = normalize(block.text)
            if not value:
                continue
            # Never assign a previous section's metadata to a new section.
            boundary = block.metadata.section != context.section
            if boundary and text:
                yield make(text, context)
                index += 1
                text = ""
            for part in self._split(value, self.config.max_tokens - self.config.overlap_tokens):
                if not text:
                    context = block.metadata
                    text = part
                    continue
                candidate = text + "\n\n" + part
                too_large = self.count(candidate) > self.config.max_tokens
                reached_target = (
                    self.count(text) >= self.config.target_tokens
                    and self.count(part) >= self.config.target_tokens // 4
                )
                if too_large or reached_target:
                    yield make(text, context)
                    index += 1
                    overlap = self._tail(text)
                    text = part
                    context = block.metadata
                    if overlap and self.count(overlap + "\n\n" + part) <= self.config.max_tokens:
                        text = overlap + "\n\n" + part
                else:
                    text = candidate
        if text:
            yield make(text, context)
