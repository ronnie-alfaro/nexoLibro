"""Recursive discovery without following symlinks or traversing hidden directories."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)
SYSTEM_NAMES = {"thumbs.db", "desktop.ini", "__macosx", "$recycle.bin", "system volume information"}


def ignored(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.startswith((".", "~", "~$"))
        or name in SYSTEM_NAMES
        or name.endswith(("~", ".tmp", ".part", ".bak", ".swp"))
    )


def discover(root: Path, extensions: list[str], recursive: bool = True) -> Iterator[Path]:
    if not root.exists():
        raise FileNotFoundError(root)

    def eligible(path: Path) -> bool:
        try:
            return (
                not ignored(path)
                and not path.is_symlink()
                and path.suffix.lower() in extensions
                and path.is_file()
                and path.stat().st_size > 0
            )
        except OSError as exc:
            logger.warning("Cannot stat %s: %s", path, exc)
            return False

    if root.is_file():
        if eligible(root):
            yield root.resolve()
        return
    for directory, dirs, files in os.walk(root, onerror=lambda e: logger.error("Discovery: %s", e)):
        dirs[:] = (
            sorted(
                d for d in dirs if not ignored(Path(d)) and not (Path(directory) / d).is_symlink()
            )
            if recursive
            else []
        )
        for name in sorted(files):
            path = Path(directory) / name
            if eligible(path):
                yield path.resolve()
