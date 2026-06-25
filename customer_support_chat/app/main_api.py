"""
Async API server for the TravelOps Swarm travel support system.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as redis
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from customer_support_chat.app.core.db import initialize_postgres_from_sqlite
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.core.redis_controls import (
    begin_idempotent_request,
    check_token_bucket_rate_limit,
    close_async_redis_client,
    complete_idempotent_request,
    distributed_semaphore,
    fail_idempotent_request,
    get_json_cache,
    make_digest,
    set_async_redis_client,
    set_json_cache,
)
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.graph import multi_agentic_graph
from customer_support_chat.app.services.utils import download_and_prepare_db


chat_semaphore = asyncio.Semaphore(get_settings().CHAT_CONCURRENCY_LIMIT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    download_and_prepare_db()
    initialize_postgres_from_sqlite()

    settings = get_settings()
    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        set_async_redis_client(redis_client)
        logger.info("Connected to Redis")
    except Exception as exc:
        logger.warning(f"Redis connection failed: {exc}. Running without Redis controls.")
        set_async_redis_client(None)

    yield

    await close_async_redis_client()


app = FastAPI(
    title="TravelOps Swarm API",
    description="High-concurrency multi-agent travel support system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    passenger_id: str = "5102 899977"


class ChatResponse(BaseModel):
    response: str
    thread_id: str


def get_cache_key(thread_id: str, message: str) -> str:
    return f"cache:chat:{thread_id}:{make_digest(message)}"


def get_request_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def get_idempotency_key(
    http_request: Request,
    chat_request: ChatRequest,
) -> str:
    explicit_key = http_request.headers.get("idempotency-key")
    if explicit_key:
        return make_digest("header", chat_request.passenger_id, explicit_key)
    thread_marker = chat_request.thread_id or "new-thread"
    return make_digest("chat", chat_request.passenger_id, thread_marker, chat_request.message)


async def cache_response(key: str, response: str, ttl: int) -> None:
    await set_json_cache(key, response, ttl)


async def get_cached_response(key: str) -> str | None:
    return await get_json_cache(key)


async def check_rate_limit(passenger_id: str, ip_address: str) -> None:
    settings = get_settings()
    dimensions = [
        (
            f"rate:chat:user:{passenger_id}",
            max(settings.CHAT_RATE_LIMIT_BURST, settings.CHAT_USER_RATE_LIMIT_PER_MINUTE),
            settings.CHAT_USER_RATE_LIMIT_PER_MINUTE,
        ),
        (
            f"rate:chat:ip:{ip_address}",
            max(settings.CHAT_RATE_LIMIT_BURST, settings.CHAT_IP_RATE_LIMIT_PER_MINUTE),
            settings.CHAT_IP_RATE_LIMIT_PER_MINUTE,
        ),
        (
            "rate:chat:global",
            max(settings.CHAT_RATE_LIMIT_BURST, settings.CHAT_GLOBAL_RATE_LIMIT_PER_MINUTE),
            settings.CHAT_GLOBAL_RATE_LIMIT_PER_MINUTE,
        ),
    ]
    allowed, blocked_key, retry_ms = await check_token_bucket_rate_limit(dimensions)
    if allowed:
        return

    retry_after = max(1, int((retry_ms + 999) / 1000))
    raise HTTPException(
        status_code=429,
        detail={
            "message": "Rate limit exceeded",
            "dimension": blocked_key,
            "retry_after_seconds": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    """Handle a chat request with Redis-backed protection."""
    idempotency_key: str | None = None
    idempotency_owned = False
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        settings = get_settings()

        await check_rate_limit(request.passenger_id, get_request_ip(http_request))

        idempotency_key = get_idempotency_key(http_request, request)
        idempotency_state, idempotency_result = await begin_idempotent_request(
            idempotency_key,
            settings.CHAT_IDEMPOTENCY_PENDING_TTL_SECONDS,
        )
        if idempotency_state == "replay":
            return ChatResponse(**idempotency_result)
        if idempotency_state == "pending":
            raise HTTPException(
                status_code=409,
                detail="Duplicate request is already being processed",
            )
        idempotency_owned = True

        cache_key = get_cache_key(thread_id, request.message)
        cached_response = await get_cached_response(cache_key)
        if cached_response:
            payload = {"response": cached_response, "thread_id": thread_id}
            await complete_idempotent_request(
                idempotency_key,
                payload,
                settings.CHAT_IDEMPOTENCY_TTL_SECONDS,
            )
            return ChatResponse(**payload)

        config = {
            "configurable": {
                "passenger_id": request.passenger_id,
                "thread_id": thread_id,
            }
        }

        def sync_chat():
            try:
                result = multi_agentic_graph.invoke(
                    {"messages": [{"role": "user", "content": request.message}]},
                    config=config,
                )
                return result["messages"][-1].content
            except Exception as exc:
                logger.error(f"Chat processing error: {exc}")
                raise

        try:
            async with chat_semaphore:
                async with distributed_semaphore(
                    "chat",
                    settings.CHAT_DISTRIBUTED_CONCURRENCY_LIMIT,
                    settings.CHAT_REQUEST_TIMEOUT_SECONDS,
                    wait_timeout_seconds=1,
                ) as acquired:
                    if not acquired:
                        raise HTTPException(
                            status_code=503,
                            detail="Chat concurrency limit reached",
                        )
                    response_content = await asyncio.wait_for(
                        asyncio.to_thread(sync_chat),
                        timeout=settings.CHAT_REQUEST_TIMEOUT_SECONDS,
                    )
        except asyncio.TimeoutError:
            logger.exception("Chat request timed out")
            if idempotency_key:
                await fail_idempotent_request(idempotency_key)
            raise HTTPException(status_code=504, detail="Chat request timed out")

        background_tasks.add_task(
            cache_response,
            cache_key,
            response_content,
            settings.CHAT_CACHE_TTL_SECONDS,
        )

        payload = {"response": response_content, "thread_id": thread_id}
        await complete_idempotent_request(
            idempotency_key,
            payload,
            settings.CHAT_IDEMPOTENCY_TTL_SECONDS,
        )
        return ChatResponse(**payload)

    except HTTPException:
        if idempotency_key and idempotency_owned:
            await fail_idempotent_request(idempotency_key)
        raise
    except Exception as exc:
        if idempotency_key:
            await fail_idempotent_request(idempotency_key)
        logger.exception("API error")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
async def get_metrics():
    return {
        "active_connections": 0,
        "total_requests": 0,
    }


if __name__ == "__main__":
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop" if "uvloop" in globals() else "asyncio",
        http="httptools",
    )
