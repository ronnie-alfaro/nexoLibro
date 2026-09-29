import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, cast

from conftest import PipelineFixture
from fastapi.testclient import TestClient

from library_ingestor.config import LlamaConfig
from library_ingestor.llm.llama_client import LlamaClient
from library_ingestor.retrieval.answer import AnswerService
from library_ingestor.retrieval.catalog import Catalog
from library_ingestor.retrieval.models import Question, Source, Turn
from library_ingestor.retrieval.retriever import Retriever
from library_ingestor.web.app import create_app


class FakeRetriever:
    def __init__(self, sources: list[Source]) -> None:
        self.sources = sources

    def retrieve(self, query: str, document_id: str | None = None) -> list[Source]:
        return list(self.sources)


class FakeLlama:
    config = LlamaConfig(context_tokens=2048, max_output_tokens=256)

    def __init__(self, answer: str = "Conocimiento documentado [1].") -> None:
        self.answer = answer
        self.calls = 0
        self.last_messages: list[dict[str, str]] = []

    async def info(self) -> dict[str, Any]:
        return {"ready": True, "model": "local", "context_tokens": 2048}

    async def count_tokens(self, messages: list[dict[str, str]]) -> int:
        self.last_messages = messages
        return 1800 if len(messages) > 2 or len(messages[-1]["content"]) > 1200 else 500

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[dict[str, Any]]:
        self.calls += 1
        yield {"type": "token", "text": self.answer}
        yield {"type": "finish", "reason": "stop"}

    async def close(self) -> None:
        pass


def source(number: int = 1) -> Source:
    return Source(
        number=number,
        point_id=str(number),
        document_id="a" * 64,
        title="Test book",
        filename="book.txt",
        chunk_index=number,
        score=0.8,
        text="Evidence. " * 25,
    )


def service(sources: list[Source], llama: FakeLlama) -> AnswerService:
    return AnswerService(cast(Retriever, FakeRetriever(sources)), cast(LlamaClient, llama))


def test_no_evidence_never_calls_llm() -> None:
    async def run() -> None:
        llama = FakeLlama()
        events = [event async for event in service([], llama).answer(Question(question="Unknown"))]
        assert llama.calls == 0 and events[-1]["generated"] is False

    asyncio.run(run())


def test_context_budget_and_citation_warning() -> None:
    async def run() -> None:
        llama = FakeLlama("Texto [1] y cita inventada [99].")
        request = Question(question="Question", history=[Turn(role="user", content="Earlier")])
        events = [
            event
            async for event in service([source(i) for i in range(1, 7)], llama).answer(request)
        ]
        final = events[-1]
        assert final["citations"] == [1] and len(final["warnings"]) == 1
        assert len(llama.last_messages) == 2
        evidence = next(event["sources"] for event in events if event["type"] == "sources")
        assert 0 < len(evidence) < 6
        assert final["prompt_tokens"] + llama.config.max_output_tokens < 2048

    asyncio.run(run())


def test_web_static_stream_validation_and_origin(setup_pipeline: PipelineFixture) -> None:
    pipeline, _, _ = setup_pipeline
    app = create_app(
        pipeline.settings,
        service([source()], FakeLlama()),
        Catalog(pipeline.settings.ingestion.state_dir),
    )
    with TestClient(app, base_url="http://127.0.0.1:8090") as client:
        home = client.get("/")
        assert home.status_code == 200 and "NexoLibro" in home.text
        assert "frame-ancestors 'none'" in home.headers["content-security-policy"]
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/answer-format.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/api/config").json()["name"] == "NexoLibro"
        assert client.get("/api/status").json()["model_ready"] is True
        assert client.post("/api/ask", json={"question": "test"}).status_code == 403
        headers = {"X-NexoLibro-Client": "local"}
        assert client.post("/api/ask", json={"question": " "}, headers=headers).status_code == 422
        assert (
            client.post(
                "/api/ask",
                json={"question": "test"},
                headers={**headers, "origin": "https://external.example"},
            ).status_code
            == 403
        )
        response = client.post("/api/ask", json={"question": "A question"}, headers=headers)
        events = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert response.status_code == 200
        assert events[-1]["type"] == "done"
        assert any(event["type"] == "sources" for event in events)
        # A completed request releases the single-inference gate.
        assert (
            client.post("/api/ask", json={"question": "Again"}, headers=headers).status_code == 200
        )
        assert client.get("/api/books").json()["total"] == 0


def test_generation_error_releases_gate(setup_pipeline: PipelineFixture) -> None:
    from library_ingestor.llm.llama_client import LlamaError

    class BrokenLlama(FakeLlama):
        async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[dict[str, Any]]:
            yield {"type": "token", "text": "Partial"}
            raise LlamaError("Lost local connection")

    pipeline, _, _ = setup_pipeline
    app = create_app(
        pipeline.settings,
        service([source()], BrokenLlama()),
        Catalog(pipeline.settings.ingestion.state_dir),
    )
    with TestClient(app, base_url="http://127.0.0.1:8090") as client:
        headers = {"X-NexoLibro-Client": "local"}
        for _ in range(2):
            response = client.post("/api/ask", json={"question": "Question"}, headers=headers)
            assert response.status_code == 200
            assert '"type": "error"' in response.text
            assert '"type": "done"' not in response.text
