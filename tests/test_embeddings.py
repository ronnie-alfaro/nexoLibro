from typing import Any

import numpy as np
import pytest
from conftest import CharacterTokenizer

from library_ingestor.config import EmbeddingConfig
from library_ingestor.embeddings.embedder import LocalEmbedder
from library_ingestor.errors import EmbeddingError


class FailingAccelerator:
    tokenizer = CharacterTokenizer()
    max_seq_length = 100
    device = "mps"
    calls = 0

    def get_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        self.calls += 1
        if self.device == "mps":
            raise RuntimeError("unsupported accelerator operator")
        return np.ones((len(texts), 3))

    def to(self, device: str) -> None:
        self.device = device


def test_cpu_fallback_reuses_loaded_model() -> None:
    embedder = LocalEmbedder(EmbeddingConfig())
    model = FailingAccelerator()
    embedder._model = model
    assert embedder.encode(["first", "second"]) == [[1.0, 1.0, 1.0]] * 2
    assert model.calls == 2 and model.device == "cpu"
    embedder.encode(["third"])
    assert embedder._model is model and model.calls == 3


def test_no_silent_truncation() -> None:
    embedder = LocalEmbedder(EmbeddingConfig())
    embedder._model = FailingAccelerator()
    with pytest.raises(EmbeddingError, match="truncation"):
        embedder.encode(["x" * 101])
