"""A single lazily loaded local SentenceTransformer per execution."""

import logging
from typing import Any, Protocol, cast

from library_ingestor.config import EmbeddingConfig
from library_ingestor.errors import EmbeddingError

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    def encode(self, text: str, **kwargs: Any) -> list[int]: ...
    def decode(self, ids: list[int], **kwargs: Any) -> str: ...


class Embedder(Protocol):
    @property
    def tokenizer(self) -> Tokenizer: ...
    @property
    def dimension(self) -> int: ...
    @property
    def max_tokens(self) -> int: ...
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.config.device
            if device == "auto":
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
            try:
                self._model = SentenceTransformer(
                    self.config.model,
                    revision=self.config.revision,
                    device=device,
                    cache_folder=str(self.config.cache_dir) if self.config.cache_dir else None,
                    local_files_only=self.config.offline,
                    trust_remote_code=False,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                raise EmbeddingError(
                    "Cannot load local embedding model. Run `library-ingestor download-models` "
                    "before offline ingestion."
                ) from exc
            logger.info("Embedding model=%s device=%s", self.config.model, device)
        return self._model

    @property
    def tokenizer(self) -> Tokenizer:
        return cast(Tokenizer, self._load().tokenizer)

    @property
    def dimension(self) -> int:
        value = self._load().get_embedding_dimension()
        if value is None:
            raise EmbeddingError("Model did not report its embedding dimension")
        return int(value)

    @property
    def max_tokens(self) -> int:
        return int(self._load().max_seq_length)

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        if any(
            len(self.tokenizer.encode(t, add_special_tokens=True)) > self.max_tokens for t in texts
        ):
            raise EmbeddingError("Input exceeds model token limit; refusing silent truncation")
        try:
            result = model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except (RuntimeError, NotImplementedError) as exc:
            if str(model.device).startswith(("mps", "cuda")):
                logger.warning("Accelerator failed; retrying this batch on CPU: %s", exc)
                model.to("cpu")
                result = model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            else:
                raise EmbeddingError("Embedding batch failed on CPU") from exc
        return cast(list[list[float]], result.tolist())
