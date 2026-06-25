# Document Intelligence System — RAG with Hybrid Search
A backend system for uploading documents and asking questions, with answers generated using retrieval-augmented generation (RAG). Built with FastAPI and deployed on Google Cloud.

## Features

- User registration and JWT-based authentication
- Document upload with deduplication and async processing (Celery + Redis)
- Hybrid search — BM25 keyword search combined with vector similarity using Reciprocal Rank Fusion
- Streaming LLM-generated answers (SSE)
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

Hybrid search was benchmarked against semantic-only search using RAGAS on a 20-question test set. The evaluation covered direct questions, theme/concept queries, cross-file queries, exact-keyword queries, and negative (out-of-domain) queries.

| Metric | Semantic only | Hybrid search |
|---|---|---|
| Faithfulness |  0.346 | 0.497 |
| Answer relevancy | 0.325 | 0.381 |
| Context recall | 0.750 | 0.780 |

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
| POST | `/api/hybrid_search` | hybrid search((BM25 + semantic + RRF + CrossEncoder)) |
| POST | `/api/search` | Semantic-only search (baseline) |
| POST | `/api/search/stream` | hybrid search with streamed answer |
| GET | `/api/search/history` | Get search history for current user |
