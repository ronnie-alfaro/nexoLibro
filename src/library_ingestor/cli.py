"""Terminal-only CLI for ingestion and diagnostic similarity search."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from filelock import FileLock, Timeout
from rich.console import Console
from rich.table import Table

from library_ingestor.config import Settings, load_settings
from library_ingestor.discovery import discover
from library_ingestor.embeddings.embedder import LocalEmbedder
from library_ingestor.errors import ConfigurationError, IngestorError
from library_ingestor.ingestion.pipeline import Pipeline, index_identity
from library_ingestor.ingestion.state import StateDB
from library_ingestor.parsers.base import ParserRegistry
from library_ingestor.parsers.docling_parser import DoclingParser
from library_ingestor.parsers.txt_parser import TxtParser
from library_ingestor.utils.hashing import file_sha256
from library_ingestor.utils.logging import configure_logging
from library_ingestor.vectorstore.qdrant_store import QdrantStore

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
console = Console(markup=False, highlight=False)
ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="YAML configuration file")]


def registry(settings: Settings) -> ParserRegistry:
    parsers = ParserRegistry()
    parsers.register(".txt", TxtParser(settings.parsing.txt_encoding))
    docling = DoclingParser(settings.parsing)
    parsers.register(".pdf", docling)
    parsers.register(".epub", docling)
    return parsers


@contextmanager
def resources(settings: Settings) -> Iterator[tuple[StateDB, QdrantStore]]:
    state = StateDB(settings.ingestion.state_dir / "state.db")
    store = QdrantStore(settings.qdrant)
    try:
        yield state, store
    finally:
        store.close()
        state.close()


@contextmanager
def errors() -> Iterator[None]:
    try:
        yield
    except typer.Exit:
        raise
    except Timeout as exc:
        console.print("Another command is using this index. Retry when it finishes.")
        raise typer.Exit(2) from exc
    except (IngestorError, OSError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(2) from exc
    except Exception as exc:
        console.print(f"Operation failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(2) from exc


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, resolve_path=True)],
    config: ConfigOption = Path("config.yaml"),
    recursive: Annotated[bool | None, typer.Option("--recursive/--no-recursive")] = None,
    dry_run: Annotated[bool, typer.Option(help="Discover/hash only; no model or Qdrant")] = False,
    force: Annotated[bool, typer.Option(help="Reprocess already indexed documents")] = False,
) -> None:
    """Parse, chunk, embed locally and index a library."""
    with errors():
        settings = load_settings(config)
        recurse = settings.ingestion.recursive if recursive is None else recursive
        if dry_run:
            # Read-only SQLite, no directories, logs, model loading, or Qdrant calls.
            import sqlite3

            db_path = settings.ingestion.state_dir / "state.db"
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) if db_path.exists() else None
            try:
                count = 0
                seen: set[str] = set()
                for file in discover(path, settings.ingestion.supported_extensions, recurse):
                    count += 1
                    digest = file_sha256(file)
                    row = (
                        db.execute(
                            "SELECT status FROM documents WHERE file_hash=?", (digest,)
                        ).fetchone()
                        if db
                        else None
                    )
                    old = (
                        db.execute(
                            "SELECT file_hash FROM paths WHERE filepath=?", (str(file),)
                        ).fetchone()
                        if db
                        else None
                    )
                    action = (
                        "FORCE"
                        if force
                        else "REINDEX"
                        if old and old[0] != digest
                        else "SKIP"
                        if row and row[0] == "completed"
                        else "INGEST"
                    )
                    if digest in seen:
                        action = "SKIP (duplicate in this discovery)"
                    seen.add(digest)
                    console.print(f"{action}: {file} ({digest[:12]})")
                console.print(f"Dry run: {count} documents; decisions based on local state only.")
            finally:
                if db:
                    db.close()
            return
        configure_logging(settings.ingestion.state_dir)
        with resources(settings) as (state, store):
            pipeline = Pipeline(
                settings,
                registry(settings),
                LocalEmbedder(settings.embedding),
                store,
                state,
                console.print,
            )
            result = pipeline.ingest(path, recurse, force)
        console.print("\nINGESTION COMPLETE")
        console.print(
            f"Documents discovered: {result.discovered}\nProcessed: {result.processed}"
            f"\nSkipped: {result.skipped}\nFailed: {result.failed}"
            f"\nChunks created: {result.chunks_created}"
        )
        seconds = int(result.elapsed_seconds)
        console.print(f"Time: {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
        if result.failures:
            console.print("Failed documents:")
            for filename, error in result.failures:
                console.print(f"- {filename}: {error}")
            raise typer.Exit(1)


@app.command()
def status(config: ConfigOption = Path("config.yaml")) -> None:
    """Summarize local state and count actual Qdrant points."""
    with errors():
        settings = load_settings(config)
        with resources(settings) as (state, store):
            console.print(f"State: {settings.ingestion.state_dir / 'state.db'}")
            console.print(f"Documents: {state.counts()}")
            console.print(f"Pending recovery transactions: {len(state.journals())}")
            console.print(f"Collection: {settings.qdrant.collection}; points: {store.count()}")


@app.command("list-books")
def list_books(config: ConfigOption = Path("config.yaml")) -> None:
    """List document identities and their ingestion status."""
    with errors():
        settings = load_settings(config)
        state = StateDB(settings.ingestion.state_dir / "state.db")
        try:
            table = Table("Title / filename", "Author", "Chunks", "Status", "Hash")
            for row in state.books():
                meta = json.loads(row["metadata"])
                table.add_row(
                    meta["title"] or meta["filename"],
                    meta["author"] or "—",
                    str(row["chunks_created"]),
                    row["status"],
                    row["file_hash"][:12],
                )
            console.print(table)
        finally:
            state.close()


def check_readable(state: StateDB, store: QdrantStore, settings: Settings) -> None:
    if state.journals():
        raise ConfigurationError("Interrupted write: run ingest to recover before reading")
    if store.identity() != index_identity(settings):
        raise ConfigurationError("Collection missing or configuration differs from its index")


@app.command()
def inspect(
    query: Annotated[str, typer.Argument(help="Title, path or hash fragment")] = "",
    config: ConfigOption = Path("config.yaml"),
    samples: Annotated[int, typer.Option(min=0, max=20)] = 3,
) -> None:
    """Show metadata, actual Qdrant counts and sample chunks."""
    with errors():
        settings = load_settings(config)
        with (
            resources(settings) as (state, store),
            FileLock(settings.ingestion.state_dir / "ingestion.lock", timeout=0),
        ):
            check_readable(state, store, settings)
            books = state.books(query)
            if not books:
                console.print("No matching documents")
            for row in books:
                meta = json.loads(row["metadata"])
                console.print(
                    f"\nTitle: {meta['title']}\nAuthor: {meta['author']}"
                    f"\nFile: {row['filepath']}\nHash: {row['file_hash']}"
                    f"\nChunks: {row['chunks_created']} (state), "
                    f"{store.count(row['file_hash'])} (Qdrant)"
                    f"\nLanguage: {meta['language']}"
                    f"\nEmbedding model: {row['embedding_model']}"
                    f"\nIndexed date: {row['ingested_at']}\nStatus: {row['status']}"
                )
                for point in islice(store.records([row["file_hash"]]), samples):
                    payload = point.payload or {}
                    console.print(
                        f"Chunk {payload.get('chunk_index')}: {str(payload.get('text', ''))[:600]}"
                    )


@app.command()
def search(
    query: str,
    config: ConfigOption = Path("config.yaml"),
    limit: Annotated[int, typer.Option(min=1, max=100)] = 5,
) -> None:
    """Diagnostic local embedding → cosine search. No LLM."""
    with errors():
        settings = load_settings(config)
        with (
            resources(settings) as (state, store),
            FileLock(settings.ingestion.state_dir / "ingestion.lock", timeout=0),
        ):
            check_readable(state, store, settings)
            embedder = LocalEmbedder(settings.embedding)
            vector = embedder.encode([query])[0]
            for point in store.search(vector, limit):
                payload = point.payload or {}
                console.print(
                    f"\nScore: {point.score:.4f}"
                    f"\nBook: {payload.get('title') or payload.get('filename')}"
                    f"\nAuthor: {payload.get('author')}"
                    f"\nChapter/section: {payload.get('chapter') or payload.get('section')}"
                    f"\n{str(payload.get('text', ''))[:700]}"
                )


@app.command("download-models")
def download_models(
    config: ConfigOption = Path("config.yaml"),
    docling: Annotated[
        bool, typer.Option("--docling", help="Also download PDF layout/table models")
    ] = False,
) -> None:
    """Explicit online setup: download public model weights, without opening documents."""
    with errors():
        settings = load_settings(config)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        settings.embedding.offline = False
        embedder = LocalEmbedder(settings.embedding)
        console.print(f"Downloading/loading {settings.embedding.model}…")
        console.print(f"Embedding model ready: {embedder.dimension} dimensions")
        if docling:
            from docling.utils.model_downloader import download_models as download_docling

            output = settings.parsing.artifacts_path or Path("models/docling").resolve()
            download_docling(
                output_dir=output,
                with_layout=True,
                with_tableformer=True,
                with_code_formula=False,
                with_picture_classifier=False,
                with_smolvlm=False,
                with_granite_vision=False,
            )
            console.print(f"Docling artifacts: {output}")


@app.command()
def serve(config: ConfigOption = Path("config.yaml")) -> None:
    """Open NexoLibro's local interface (connects to an existing llama-server)."""
    with errors():
        import uvicorn

        from library_ingestor.web.app import create_app

        settings = load_settings(config)
        configure_logging(settings.ingestion.state_dir)
        console.print(f"{settings.branding.name}: http://{settings.web.host}:{settings.web.port}")
        console.print(f"Modelo local: {settings.llama.base_url}")
        uvicorn.run(
            create_app(settings),
            host=settings.web.host,
            port=settings.web.port,
            workers=1,
            proxy_headers=False,
            access_log=False,
        )


@app.command("model-server")
def model_server(
    model: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True)],
    config: ConfigOption = Path("config.yaml"),
) -> None:
    """Keep a local GGUF loaded in llama.cpp, with one slot and a bounded context."""
    with errors():
        from library_ingestor.llm.runner import run_server

        settings = load_settings(config)
        console.print(f"llama.cpp: {settings.llama.base_url} · {model.name}")
        raise typer.Exit(run_server(model, settings.llama))
