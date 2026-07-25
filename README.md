# AI Research Assistant

A production-quality AI Research Assistant backend implementing document ingestion and
retrieval-augmented generation (RAG) over PDF, DOCX, TXT, and Markdown files.

## Stack

- Python 3.13, FastAPI, Pydantic Settings
- PostgreSQL + SQLAlchemy 2.x (async) + Alembic
- ChromaDB for vector storage
- LangChain + OpenAI-compatible LLM interface (works with OpenAI or any compatible gateway)
- Docker / docker-compose / GitHub Actions
- Ruff, Black, MyPy, Pytest, coverage (100% enforced)

## Project layout

```
src/ai_research_assistant/
  core/          # config, logging, exceptions, security
  db/            # SQLAlchemy engine/session, ORM models
  schemas/       # Pydantic request/response models
  repositories/  # data access layer
  services/      # ingestion, embeddings, vectorstore, retrieval, llm, memory
  api/           # FastAPI routers and dependency injection
  main.py        # application entrypoint
alembic/         # database migrations
tests/           # pytest suite
```

## Getting started

1. Copy environment variables:

   ```bash
   cp .env.example .env
   ```

2. Start all services (API, PostgreSQL, ChromaDB):

   ```bash
   docker compose up --build
   ```

3. Run database migrations:

   ```bash
   docker compose exec api alembic upgrade head
   ```

4. Open the interactive API docs at `http://localhost:8000/docs`.

## Local development (without Docker)

```bash
pip install -e ".[dev]"
alembic upgrade head
uvicorn ai_research_assistant.main:app --reload
```

## API overview

| Method | Path                                        | Description                                 |
|--------|----------------------------------------------|----------------------------------------------|
| GET    | `/api/v1/health`                             | Liveness + database connectivity              |
| GET    | `/api/v1/version`                            | Application name/version/environment          |
| POST   | `/api/v1/auth/register`                      | Register a user                               |
| POST   | `/api/v1/auth/login`                         | Obtain a bearer access token                  |
| POST   | `/api/v1/documents`                          | Upload a document for ingestion               |
| GET    | `/api/v1/documents`                          | List the current user's documents             |
| POST   | `/api/v1/documents/search`                   | Semantic search over ingested chunks          |
| GET    | `/api/v1/documents/{id}`                     | Get a document's status/metadata              |
| DELETE | `/api/v1/documents/{id}`                     | Delete a document and its embeddings          |
| POST   | `/api/v1/chat/sessions`                      | Create a chat session                         |
| GET    | `/api/v1/chat/sessions`                      | List chat sessions                            |
| GET    | `/api/v1/chat/sessions/{id}`                 | Get a session with full message history       |
| DELETE | `/api/v1/chat/sessions/{id}`                 | Delete a session and its messages             |
| GET    | `/api/v1/chat/sessions/{id}/messages`        | Get conversation history (with citations)     |
| POST   | `/api/v1/chat/sessions/{id}/query`           | Ask a question (RAG over documents)           |
| POST   | `/api/v1/chat/sessions/{id}/query/stream`    | Same as above, streamed via Server-Sent Events|

All document/chat endpoints require `Authorization: Bearer <token>`. Full interactive docs
(OpenAPI/Swagger) are served at `/docs`, ReDoc at `/redoc`.

Chat responses (and persisted assistant messages) include `sources` (per-chunk citations with
document id, title, chunk index, and similarity score) and `metadata` (model name, latency,
retrieved chunk count, generation timestamp) for frontend rendering and reference tracking.

## Ingestion pipeline

Upload → text extraction (pypdf / python-docx / plain text) → chunking
(`RecursiveCharacterTextSplitter`) → embedding generation (OpenAI-compatible embeddings
endpoint) → vectors stored in ChromaDB, chunk metadata stored in PostgreSQL.

## Retrieval pipeline

Query → embed → ChromaDB similarity search (cosine, scoped per user) → context builder
assembles a citation-annotated context block → prompt builder composes a grounded system
prompt → conversation memory supplies bounded chat history → LLM generates a cited answer.

## Quality tooling

```bash
ruff check .
black --check .
mypy .
pytest   # runs with coverage; fails if total coverage drops below 100%
```

Coverage reports are written to `coverage.xml` (and terminal `--cov-report=term-missing`).

## Continuous integration

`.github/workflows/ci.yml` runs on every push/PR:

- `lint`: ruff, black --check, mypy
- `test`: pytest + coverage against a real PostgreSQL service container, plus
  `alembic upgrade head` / `alembic check` to catch migration drift
- `docker-build`: builds the production Docker image

## Database migrations

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```
