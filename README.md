# Document Intelligence System — RAG with Hybrid Search
A backend system for uploading documents and asking questions, with answers generated using retrieval-augmented generation (RAG). Built with FastAPI and deployed on Google Cloud.

## Features

- User registration and JWT-based authentication
- Document upload with deduplication and async processing (Celery + Redis)
- Hybrid search — BM25 keyword search combined with vector similarity using Reciprocal Rank Fusion
- Streaming LLM-generated answers (SSE)
- Multi-turn conversations — follow-up questions are condensed into standalone queries using recent turn history, with coherent, context-aware answers
- List and delete uploaded documents
- Retrieval quality benchmarked with RAGAS

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI (async) |
| Database | PostgreSQL (async SQLAlchemy) |
| Vector store | Chroma |
| Task queue | Celery + Redis |
| LLM | Groq |
| Embeddings | sentence-transformers |
| Auth | FastAPI-Users (JWT) |
| Deployment | Docker Compose, Nginx reverse proxy, Google Cloud |


## Retrieval evaluation

Hybrid search was benchmarked against semantic-only search using RAGAS on a 69-question test set spanning multiple document types (`test/eval_file.json`):

- **Literary/fiction documents** — character, plot, and relationship questions
- **Technical documents** — language/syntax and database concept questions

The evaluation covered direct questions, theme/concept queries, cross-file queries, exact-keyword queries, and negative (out-of-domain) queries.

| Metric               | Semantic |   Hybrid |
|----------------------|----------|----------|
| faithfulness         |   0.8482 |   0.8203 |
| answer_relevancy     |   0.6580 |   0.6343 |
| context_recall       |   0.5531 |   0.5369 |

## Running locally

```bash
cd document-analyzer
cp .env.example .env
docker compose -f docker-compose_prod.yml up
```

API will be available at `http://localhost:8080/docs`.

## API

| Endpoint | Method | Description |
|---|---|---|
| POST | `/api/auth/register` | Register a new user |
| POST | `/api/auth/login` | Log in and receive a JWT |
| POST | `/api/files/hybrid_upload` | Upload a document, indexing happens async via celery |
| GET | `/api/files/list` | List uploaded documents |
| DELETE | `/api/files/{file_id}` | Delete a document |
| POST | `/api/hybrid_search` | hybrid search((BM25 + semantic + RRF + CrossEncoder)), optional `conversation_id` for multi-turn follow-ups |
| POST | `/api/search` | Semantic-only search (baseline) |
| POST | `/api/search/stream` | hybrid search with streamed answer, optional `conversation_id` for multi-turn follow-ups |
| GET | `/api/search/history` | Get search history for current user |
| POST | `/api/conversations` | Start a new multi-turn conversation |
| GET | `/api/conversations` | List the current user's conversations |
| GET | `/api/conversations/{conversation_id}` | Get a conversation and its ordered turns |
| DELETE | `/api/conversations/{conversation_id}` | Delete a conversation (turns are detached, not deleted) |
