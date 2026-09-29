"""Public, actionable error types."""


class IngestorError(Exception):
    """Base error exposed by the CLI."""


class ParsingError(IngestorError):
    """A document cannot be parsed without losing content."""


class ConfigurationError(IngestorError):
    """Invalid or incompatible configuration."""


class StoreError(IngestorError):
    """Storage operation failed."""


class RecoveryError(StoreError):
    """A pending transaction requires recovery before any further writes."""


class EmbeddingError(IngestorError):
    """Local model unavailable or encoding failed."""
