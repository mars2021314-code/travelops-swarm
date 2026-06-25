# TravelOps Swarm

Multi-agent RAG travel support demo for booking and service workflows. The app
uses LangGraph for agent routing, FastAPI for the HTTP API, Qdrant for vector
search, Redis for distributed controls, and Postgres or SQLite for runtime
state.

## Features

- Multi-agent support flow for flight, hotel, car rental, and excursion tasks.
- FastAPI chat API with Gunicorn/Uvicorn deployment.
- Redis token-bucket rate limiting at user, IP, and global levels.
- Redis-backed distributed concurrency control.
- Redis request deduplication and idempotency protection.
- Redis caching for high-frequency read queries and chat responses.
- Redis distributed locks for sensitive write operations.
- Optional Postgres-backed LangGraph checkpointing.
- Docker Compose stack with Nginx, Redis, Postgres, Qdrant, and API replicas.

## Architecture

```text
Client
  -> Nginx
  -> FastAPI / LangGraph workers
      -> Redis: rate limits, idempotency, cache, locks
      -> Qdrant: vector retrieval
      -> Postgres or SQLite: runtime travel data and checkpoints
```

## Requirements

- Python 3.12
- Poetry
- Docker and Docker Compose
- DeepSeek API key, or compatible OpenAI-style chat endpoint
- Optional OpenAI API key for embedding workflows

## Configuration

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Then fill in the required values:

```env
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

Never commit `.env`, `.dev.env`, database files, Qdrant storage, or API keys.

## Run With Docker

```bash
docker compose up --build -d
```

Check service health:

```bash
curl http://localhost/health
```

Stop the stack:

```bash
docker compose down
```

## Run Locally

Install dependencies:

```bash
poetry install
```

Start local dependencies as needed, then run the API:

```bash
poetry run python -m customer_support_chat.app.main_api
```

## Chat API

Send a chat message:

```bash
curl -X POST "http://localhost/chat" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: demo-chat-001" \
  -d '{
    "message": "Hello, please briefly tell me my flight info.",
    "thread_id": "demo-thread-001",
    "passenger_id": "5102 899977"
  }'
```

Continue the same conversation by keeping `thread_id` unchanged and sending a
new `Idempotency-Key` for each request.

## Important Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | Primary chat model API key | empty |
| `DEEPSEEK_BASE_URL` | OpenAI-compatible chat endpoint | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | Chat model name | `deepseek-chat` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `QDRANT_URL` | Qdrant HTTP endpoint | `http://localhost:6333` |
| `DATABASE_URL` | Optional Postgres runtime database | empty |
| `CHECKPOINT_DATABASE_URL` | Optional Postgres checkpoint database | `DATABASE_URL` |
| `CHAT_USER_RATE_LIMIT_PER_MINUTE` | User-level token refill rate | `60` |
| `CHAT_IP_RATE_LIMIT_PER_MINUTE` | IP-level token refill rate | `120` |
| `CHAT_GLOBAL_RATE_LIMIT_PER_MINUTE` | Global token refill rate | `600` |
| `CHAT_DISTRIBUTED_CONCURRENCY_LIMIT` | Cross-instance chat concurrency | `CHAT_CONCURRENCY_LIMIT` |
| `CHAT_IDEMPOTENCY_TTL_SECONDS` | Completed idempotency result TTL | `3600` |
| `REDIS_QUERY_CACHE_TTL_SECONDS` | Read-query cache TTL | `300` |

## Tests

Run the unit tests:

```bash
poetry run python -m unittest
```

Run a compile check:

```bash
poetry run python -m compileall customer_support_chat vectorizer
```

## Open Source Hygiene

This repository intentionally ignores generated and local runtime data:

- `.env` and `.dev.env`
- SQLite runtime databases
- Qdrant local storage
- Redis and Postgres Docker volumes
- Python caches and test temp directories

If a secret was committed before the repository was public, rotate it before
publishing and remove it from Git history.

## License

MIT. See [LICENSE](LICENSE).
