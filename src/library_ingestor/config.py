"""Validated YAML settings with LIBRARY_...__... environment overrides."""

import os
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

from library_ingestor.errors import ConfigurationError


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QdrantConfig(Options):
    host: Literal["localhost", "127.0.0.1", "::1"] = "127.0.0.1"
    port: int = Field(default=6333, ge=1, le=65535)
    collection: str = "library"
    batch_size: int = Field(default=64, ge=1)
    timeout: int = Field(default=60, ge=1)


class EmbeddingConfig(Options):
    model: str = "BAAI/bge-m3"
    revision: str | None = None
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    offline: bool = True
    cache_dir: Path | None = None


class ChunkingConfig(Options):
    target_tokens: int = Field(default=700, ge=8)
    max_tokens: int = Field(default=1000, ge=8)
    overlap_tokens: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_sizes(self) -> Self:
        if not self.overlap_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("Require overlap_tokens < target_tokens <= max_tokens")
        return self


class IngestionConfig(Options):
    recursive: bool = True
    supported_extensions: list[str] = [".epub", ".pdf", ".txt"]
    state_dir: Path = Path(".library_ingestor")

    @field_validator("supported_extensions")
    @classmethod
    def extensions(cls, value: list[str]) -> list[str]:
        return ["." + item.lower().lstrip(".") for item in value]


class ParsingConfig(Options):
    ocr: Literal["never", "auto"] = "never"
    artifacts_path: Path | None = None
    txt_encoding: str = "utf-8-sig"


class LlamaConfig(Options):
    host: Literal["localhost", "127.0.0.1", "::1"] = "127.0.0.1"
    port: int = Field(default=8081, ge=1, le=65535)
    model: str = "local-library"
    context_tokens: int = Field(default=8192, ge=2048, le=131072)
    max_output_tokens: int = Field(default=768, ge=64, le=4096)
    temperature: float = Field(default=0.2, ge=0, le=2)
    timeout: float = Field(default=180, ge=1)

    @model_validator(mode="after")
    def output_fits(self) -> Self:
        if self.max_output_tokens + 512 >= self.context_tokens:
            raise ValueError("Leave at least 512 context tokens for the prompt")
        return self

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


class RetrievalConfig(Options):
    candidate_limit: int = Field(default=24, ge=1, le=100)
    source_limit: int = Field(default=6, ge=1, le=12)
    max_per_book: int = Field(default=2, ge=1, le=12)
    min_score: float = Field(default=0.35, ge=-1, le=1)
    embedding_cache_size: int = Field(default=128, ge=0, le=1024)


class WebConfig(Options):
    host: Literal["localhost", "127.0.0.1", "::1"] = "127.0.0.1"
    port: int = Field(default=8090, ge=1, le=65535)


class BrandingConfig(Options):
    """Product copy shown by the local interface; no content is stored here."""

    name: str = Field(default="NexoLibro", min_length=1, max_length=60)
    tagline: str = Field(default="Tu biblioteca local", min_length=1, max_length=120)
    assistant_name: str = Field(default="Asistente", min_length=1, max_length=60)
    welcome_title: str = Field(default="Explora tu biblioteca", min_length=1, max_length=100)
    welcome_description: str = Field(
        default="Encuentra respuestas en tus propios documentos, siempre en tu dispositivo.",
        min_length=1,
        max_length=240,
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LIBRARY_", env_nested_delimiter="__", extra="forbid"
    )
    qdrant: QdrantConfig = QdrantConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    ingestion: IngestionConfig = IngestionConfig()
    parsing: ParsingConfig = ParsingConfig()
    llama: LlamaConfig = LlamaConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    web: WebConfig = WebConfig()
    branding: BrandingConfig = BrandingConfig()


def load_settings(path: Path = Path("config.yaml")) -> Settings:
    """Resolve filesystem options relative to the configuration file, not the shell."""
    try:
        values = yaml.safe_load(path.read_text()) if path.exists() else {}
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a mapping")
        # Environment has priority over YAML, including nested partial overrides.
        env = EnvSettingsSource(Settings)()
        for key, value in env.items():
            if isinstance(value, dict) and isinstance(values.get(key), dict):
                values[key].update(value)
            else:
                values[key] = value
        settings = Settings(**values)
        base = path.resolve().parent
        settings.ingestion.state_dir = (base / settings.ingestion.state_dir).resolve()
        if settings.embedding.cache_dir:
            settings.embedding.cache_dir = (base / settings.embedding.cache_dir).resolve()
        if settings.parsing.artifacts_path:
            settings.parsing.artifacts_path = (base / settings.parsing.artifacts_path).resolve()
        if settings.embedding.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return settings
    except (ValueError, OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load {path}: {exc}") from exc
