# NexoLibro

NexoLibro es una aplicación local y configurable para indexar una biblioteca EPUB, PDF o TXT, buscarla semánticamente y consultarla con un modelo local. Tus documentos no salen de tu equipo: Qdrant, los embeddings y llama.cpp se ejecutan localmente.

## Qué incluye

- Ingestión incremental con SHA-256, SQLite y recuperación ante interrupciones.
- Metadata de EPUB/PDF, fragmentación estructural y embeddings locales BGE-M3.
- Catálogo y consultas con fuentes verificables, servidos en `127.0.0.1`.
- Interfaz limpia, adaptable y sin dependencias de CDN o cuentas externas.
- Configuración YAML y overrides de entorno con `LIBRARY_`.

## Topología local

NexoLibro requiere Qdrant en Docker (`127.0.0.1:6333`). Las consultas también requieren un `llama-server` local (`127.0.0.1:8088` por defecto). No se inicia ningún modelo automáticamente y la ingestión no necesita un LLM.

```text
Documentos → parser → fragmentos → embeddings locales → Qdrant
                                                    ↓
NexoLibro web ← llama-server local ← fuentes recuperadas
```

## Inicio rápido

```bash
uv sync --locked
docker compose up -d
uv run library-ingestor download-models --docling  # descarga pesos públicos, no documentos
uv run library-ingestor ingest /ruta/a/tu/biblioteca
uv run library-ingestor model-server /ruta/a/modelo.gguf
uv run library-ingestor serve
```

Abre `http://127.0.0.1:8090`. Para comprobar archivos sin procesarlos usa `uv run library-ingestor ingest /ruta/a/tu/biblioteca --dry-run`.

## Configuración

Todo el comportamiento operativo se configura en `config.yaml`: Qdrant, embeddings, fragmentación, OCR, formatos aceptados, estado local, llama.cpp, recuperación, host/puerto y la identidad visible de la aplicación.

```yaml
branding:
  name: NexoLibro
  tagline: Tu biblioteca local
  assistant_name: Asistente
  welcome_title: Explora tu biblioteca
  welcome_description: Encuentra respuestas en tus propios documentos, siempre en tu dispositivo.
```

Los overrides tienen prioridad sobre YAML: `LIBRARY_WEB__PORT=8091`, `LIBRARY_RETRIEVAL__SOURCE_LIMIT=4` y `LIBRARY_BRANDING__NAME=MiBiblioteca`.

Las rutas son relativas al archivo de configuración. Los directorios de estado, índice, modelos, cachés, `.env` y distribuciones están excluidos de Git para evitar publicar documentos, vectores, rutas locales o secretos.

## Calidad

```bash
uv run ruff check .
uv run mypy src
uv run pytest
node --test tests/frontend/answer-format.test.mjs
```

Las GitHub Actions ejecutan estas verificaciones en cada push y pull request.
