"""Explicit foreground launch of an installed llama-server and a local GGUF."""

import shutil
import subprocess
from pathlib import Path

from library_ingestor.config import LlamaConfig
from library_ingestor.errors import ConfigurationError


def server_command(model: Path, config: LlamaConfig, executable: str = "llama-server") -> list[str]:
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise ConfigurationError("Indica un archivo GGUF local existente.")
    binary = shutil.which(executable)
    if binary is None:
        raise ConfigurationError("No se encontró llama-server en PATH. Instala llama.cpp primero.")
    return [
        binary,
        "--model",
        str(model.resolve()),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--alias",
        config.model,
        "--ctx-size",
        str(config.context_tokens),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "99",
        "--flash-attn",
        "on",
        "--cache-ram",
        "256",
        "--jinja",
        "--reasoning",
        "off",
        "--no-ui",
        "--no-context-shift",
    ]


def run_server(model: Path, config: LlamaConfig) -> int:
    try:
        return subprocess.call(server_command(model, config))
    except KeyboardInterrupt:
        return 130
