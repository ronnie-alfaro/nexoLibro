"""Local-only library ingestion."""

import os

# Set before importing transformers, Hugging Face or Docling.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
