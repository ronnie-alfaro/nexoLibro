"""File logs contain diagnostics; Rich owns terminal progress."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(state_dir / "ingestion.log", maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger("library_ingestor").setLevel(logging.INFO)
    logging.getLogger("library_ingestor").addHandler(handler)
