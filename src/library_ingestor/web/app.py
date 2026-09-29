"""One local ASGI process serves the interface and streams grounded answers."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from library_ingestor.config import Settings
from library_ingestor.embeddings.embedder import LocalEmbedder
from library_ingestor.errors import IngestorError
from library_ingestor.llm.llama_client import LlamaClient
from library_ingestor.retrieval.answer import AnswerService
from library_ingestor.retrieval.catalog import Catalog
from library_ingestor.retrieval.models import Question
from library_ingestor.retrieval.retriever import Retriever
from library_ingestor.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


def create_app(
    settings: Settings, service: AnswerService | None = None, catalog: Catalog | None = None
) -> FastAPI:
    catalog = catalog or Catalog(settings.ingestion.state_dir)
    llama = service.llama if service else LlamaClient(settings.llama)
    store = None if service else QdrantStore(settings.qdrant)
    if service is None:
        assert store is not None
        retriever = Retriever(settings, LocalEmbedder(settings.embedding), store, catalog)
        service = AnswerService(retriever, llama)
    answer_service = service
    gate = asyncio.Lock()
    status_cache: dict[str, Any] = {}
    status_time = 0.0

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await llama.close()
            if store:
                store.close()

    app = FastAPI(
        title=f"{settings.branding.name} · {settings.branding.tagline}",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def local_only(request: Request, call_next: Any) -> Response:
        if request.method == "POST":
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            valid_client = request.headers.get("x-nexolibro-client") == "local"
            if (origin and origin != expected) or not valid_client:
                return JSONResponse({"detail": "Origen no permitido"}, status_code=403)
            if int(request.headers.get("content-length", "0")) > 60000:
                return JSONResponse({"detail": "Solicitud demasiado grande"}, status_code=413)
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def home() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        nonlocal status_cache, status_time
        if time.monotonic() - status_time < 10 and status_cache:
            return {**status_cache, "busy": gate.locked()}
        result: dict[str, Any] = {
            "books": 0,
            "chunks": 0,
            "local": True,
            "model_ready": False,
            "collection": settings.qdrant.collection,
        }
        try:
            result.update(await asyncio.to_thread(catalog.stats))
        except (IngestorError, OSError) as exc:
            result["index_error"] = str(exc)
        try:
            info = await llama.info()
            result.update(model_ready=True, model=info["model"])
        except IngestorError as exc:
            result["model_error"] = str(exc)
        status_cache, status_time = result, time.monotonic()
        return {**result, "busy": gate.locked()}

    @app.get("/api/config")
    async def config() -> dict[str, str]:
        return settings.branding.model_dump()

    @app.get("/api/books")
    async def books(
        q: str = Query(default="", max_length=200),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=24, ge=1, le=60),
    ) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(catalog.books, q, offset, limit)
        except IngestorError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/ask")
    async def ask(question: Question) -> StreamingResponse:
        if gate.locked():
            raise HTTPException(
                status_code=409, detail="Hay una consulta en curso. Espera o detenla."
            )
        await gate.acquire()

        async def events() -> AsyncIterator[str]:
            try:
                async for event in answer_service.answer(question):
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            except (IngestorError, httpx.HTTPError) as exc:
                yield "data: " + json.dumps({"type": "error", "message": str(exc)}) + "\n\n"
            except Exception:
                logger.exception("Local answer failed")
                yield 'data: {"type":"error","message":"No se pudo completar la consulta."}\n\n'
            finally:
                gate.release()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
