"""Persistent HTTP client to a loopback-only llama.cpp server; no model reloads."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from library_ingestor.config import LlamaConfig
from library_ingestor.errors import IngestorError


class LlamaError(IngestorError):
    """Local inference is unavailable or returned an invalid response."""


class LlamaClient:
    def __init__(self, config: LlamaConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.client = client or httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout, connect=3),
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def _get(self, path: str) -> dict[str, Any]:
        response = await self.client.get(path, timeout=3)
        response.raise_for_status()
        value: dict[str, Any] = response.json()
        return value

    async def info(self) -> dict[str, Any]:
        try:
            props, models = await asyncio.gather(self._get("/props"), self._get("/v1/models"))
            model_list = models.get("data", [])
            if not model_list:
                raise LlamaError("llama-server no tiene un modelo cargado.")
            context = int(
                props.get("default_generation_settings", {}).get(
                    "n_ctx", self.config.context_tokens
                )
            )
            return {
                "ready": True,
                "model": model_list[0]["id"],
                "context_tokens": min(context, self.config.context_tokens),
            }
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise LlamaError(
                f"No se puede acceder a llama-server en {self.config.base_url}. "
                "Inicia el modelo local siguiendo el README."
            ) from exc

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Use the actual model's chat-template tokenizer, never the BGE tokenizer."""
        try:
            response = await self.client.post(
                "/v1/chat/completions/input_tokens", json=self._body(messages)
            )
            if response.status_code in (404, 405):
                template = await self.client.post("/apply-template", json=self._body(messages))
                template.raise_for_status()
                tokens = await self.client.post(
                    "/tokenize",
                    json={
                        "content": template.json()["prompt"],
                        "add_special": True,
                        "parse_special": True,
                    },
                )
                tokens.raise_for_status()
                return len(tokens.json()["tokens"])
            response.raise_for_status()
            return int(response.json()["input_tokens"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise LlamaError(
                "No se pudo verificar el presupuesto de contexto de llama.cpp."
            ) from exc

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[dict[str, Any]]:
        body = {
            **self._body(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache_prompt": True,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=body) as response:
                response.raise_for_status()
                finished = False
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        finished = True
                        break
                    value = json.loads(data)
                    if "error" in value:
                        raise LlamaError("llama.cpp notificó un error durante la generación.")
                    for choice in value.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if content:
                            yield {"type": "token", "text": content}
                        if choice.get("finish_reason"):
                            yield {"type": "finish", "reason": choice["finish_reason"]}
                    if value.get("usage"):
                        yield {"type": "usage", "usage": value["usage"]}
                if not finished:
                    raise LlamaError(
                        "La conexión con el modelo terminó antes de completar la respuesta."
                    )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise LlamaError(
                "Falló la generación local. Comprueba llama-server y vuelve a intentar."
            ) from exc
