import asyncio
import json
from pathlib import Path

import httpx
import pytest

from library_ingestor.config import LlamaConfig
from library_ingestor.llm.llama_client import LlamaClient, LlamaError


def test_stream_uses_local_connection_and_reports_usage() -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        bodies.append(json.loads(request.content))
        frames = [
            {"choices": [{"delta": {"content": "Respuesta [1]."}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 9},
            },
        ]
        body = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    async def run() -> None:
        client = httpx.AsyncClient(
            base_url="http://127.0.0.1:8081", transport=httpx.MockTransport(handler)
        )
        llama = LlamaClient(LlamaConfig(), client)
        events = [event async for event in llama.stream([{"role": "user", "content": "Pregunta"}])]
        assert events[0] == {"type": "token", "text": "Respuesta [1]."}
        assert events[-1]["usage"]["completion_tokens"] == 9
        assert bodies[0]["cache_prompt"] is True and bodies[0]["max_tokens"] == 768
        await llama.close()

    asyncio.run(run())


def test_count_tokens_falls_back_to_actual_chat_template() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(404)
        if request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "formatted prompt"})
        assert json.loads(request.content)["content"] == "formatted prompt"
        return httpx.Response(200, json={"tokens": [1, 2, 3]})

    async def run() -> None:
        client = httpx.AsyncClient(
            base_url="http://127.0.0.1:8081", transport=httpx.MockTransport(handler)
        )
        llama = LlamaClient(LlamaConfig(), client)
        assert await llama.count_tokens([{"role": "user", "content": "Hi"}]) == 3
        assert calls == ["/v1/chat/completions/input_tokens", "/apply-template", "/tokenize"]
        await llama.close()

    asyncio.run(run())


def test_interrupted_stream_is_not_success() -> None:
    async def run() -> None:
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200, text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            )
        )
        client = httpx.AsyncClient(base_url="http://127.0.0.1:8081", transport=transport)
        llama = LlamaClient(LlamaConfig(), client)
        with pytest.raises(LlamaError, match="terminó antes"):
            _ = [event async for event in llama.stream([])]
        await llama.close()

    asyncio.run(run())


def test_model_runner_uses_local_file_and_bounded_resources(tmp_path: Path) -> None:
    from library_ingestor.llm.runner import server_command

    model = tmp_path / "a model; with spaces.gguf"
    model.write_bytes(b"GGUF")
    command = server_command(model, LlamaConfig(), executable="true")
    assert command[command.index("--model") + 1] == str(model)
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--ctx-size") + 1] == "8192"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--no-context-shift" in command
