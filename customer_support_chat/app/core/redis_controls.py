from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Callable, Iterator

import redis
import redis.asyncio as async_redis

from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.core.settings import get_settings


_async_client: async_redis.Redis | None = None
_sync_client: redis.Redis | None = None


TOKEN_BUCKET_SCRIPT = """
local now = tonumber(ARGV[1])
local cost = tonumber(ARGV[2])
local count = tonumber(ARGV[3])

for i = 1, count do
  local capacity = tonumber(ARGV[3 + ((i - 1) * 2) + 1])
  local refill_per_ms = tonumber(ARGV[3 + ((i - 1) * 2) + 2])
  local data = redis.call("HMGET", KEYS[i], "tokens", "updated_at")
  local tokens = tonumber(data[1])
  local updated_at = tonumber(data[2])

  if tokens == nil then
    tokens = capacity
    updated_at = now
  end

  local elapsed = math.max(0, now - updated_at)
  tokens = math.min(capacity, tokens + (elapsed * refill_per_ms))

  if tokens < cost then
    local retry_ms = math.ceil((cost - tokens) / refill_per_ms)
    return {0, KEYS[i], retry_ms}
  end
end

for i = 1, count do
  local capacity = tonumber(ARGV[3 + ((i - 1) * 2) + 1])
  local refill_per_ms = tonumber(ARGV[3 + ((i - 1) * 2) + 2])
  local data = redis.call("HMGET", KEYS[i], "tokens", "updated_at")
  local tokens = tonumber(data[1])
  local updated_at = tonumber(data[2])

  if tokens == nil then
    tokens = capacity
    updated_at = now
  end

  local elapsed = math.max(0, now - updated_at)
  tokens = math.min(capacity, tokens + (elapsed * refill_per_ms)) - cost
  redis.call("HSET", KEYS[i], "tokens", tokens, "updated_at", now)
  redis.call("PEXPIRE", KEYS[i], math.ceil((capacity / refill_per_ms) * 2))
end

return {1, "", 0}
"""


SEMAPHORE_ACQUIRE_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local token = ARGV[2]
local limit = tonumber(ARGV[3])
local ttl_ms = tonumber(ARGV[4])

redis.call("ZREMRANGEBYSCORE", key, 0, now - ttl_ms)
if redis.call("ZCARD", key) < limit then
  redis.call("ZADD", key, now, token)
  redis.call("PEXPIRE", key, ttl_ms)
  return 1
end
return 0
"""


def set_async_redis_client(client: async_redis.Redis | None) -> None:
    global _async_client
    _async_client = client


def get_async_redis_client() -> async_redis.Redis | None:
    return _async_client


def get_sync_redis_client() -> redis.Redis | None:
    global _sync_client
    if _sync_client is not None:
        return _sync_client

    settings = get_settings()
    try:
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        _sync_client = client
        return client
    except Exception as exc:
        logger.warning(f"Redis sync client unavailable: {exc}")
        return None


async def close_async_redis_client() -> None:
    global _async_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None


def make_digest(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def check_token_bucket_rate_limit(
    dimensions: list[tuple[str, int, int]],
    cost: int = 1,
) -> tuple[bool, str, int]:
    client = get_async_redis_client()
    if not client:
        return True, "", 0

    active_dimensions = [
        (key, capacity, refill_per_minute)
        for key, capacity, refill_per_minute in dimensions
        if capacity > 0 and refill_per_minute > 0
    ]
    if not active_dimensions:
        return True, "", 0

    now_ms = int(time.time() * 1000)
    keys = [key for key, _, _ in active_dimensions]
    args: list[Any] = [now_ms, cost, len(active_dimensions)]
    for _, capacity, refill_per_minute in active_dimensions:
        args.extend([capacity, refill_per_minute / 60000])

    allowed, blocked_key, retry_ms = await client.eval(
        TOKEN_BUCKET_SCRIPT,
        len(keys),
        *keys,
        *args,
    )
    return bool(allowed), str(blocked_key), int(retry_ms)


@asynccontextmanager
async def distributed_semaphore(
    name: str,
    limit: int,
    ttl_seconds: int,
    wait_timeout_seconds: float = 0,
) -> AsyncIterator[bool]:
    client = get_async_redis_client()
    if not client or limit <= 0:
        yield True
        return

    key = f"semaphore:{name}"
    token = str(uuid.uuid4())
    deadline = time.monotonic() + wait_timeout_seconds
    acquired = False
    ttl_ms = max(1000, ttl_seconds * 1000)

    try:
        while True:
            now_ms = int(time.time() * 1000)
            acquired = bool(
                await client.eval(
                    SEMAPHORE_ACQUIRE_SCRIPT,
                    1,
                    key,
                    now_ms,
                    token,
                    limit,
                    ttl_ms,
                )
            )
            if acquired or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.05)
        yield acquired
    finally:
        if acquired:
            await client.zrem(key, token)


async def get_json_cache(key: str) -> Any | None:
    client = get_async_redis_client()
    if not client:
        return None
    value = await client.get(key)
    if value is None:
        return None
    return json.loads(value)


async def set_json_cache(key: str, value: Any, ttl_seconds: int) -> None:
    client = get_async_redis_client()
    if client and ttl_seconds > 0:
        await client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=True, default=str))


async def begin_idempotent_request(
    key: str,
    pending_ttl_seconds: int,
) -> tuple[str, Any | None]:
    client = get_async_redis_client()
    if not client:
        return "owned", None

    result_key = f"idempotency:result:{key}"
    pending_key = f"idempotency:pending:{key}"
    cached = await get_json_cache(result_key)
    if cached is not None:
        return "replay", cached

    acquired = await client.set(pending_key, "1", ex=pending_ttl_seconds, nx=True)
    if acquired:
        return "owned", None

    cached = await get_json_cache(result_key)
    if cached is not None:
        return "replay", cached
    return "pending", None


async def complete_idempotent_request(key: str, value: Any, ttl_seconds: int) -> None:
    client = get_async_redis_client()
    if not client:
        return
    await set_json_cache(f"idempotency:result:{key}", value, ttl_seconds)
    await client.delete(f"idempotency:pending:{key}")


async def fail_idempotent_request(key: str) -> None:
    client = get_async_redis_client()
    if client:
        await client.delete(f"idempotency:pending:{key}")


def redis_cached(ttl_seconds_attr: str = "REDIS_QUERY_CACHE_TTL_SECONDS") -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = get_settings()
            ttl_seconds = int(getattr(settings, ttl_seconds_attr, 0))
            client = get_sync_redis_client()
            if not client or ttl_seconds <= 0:
                return func(*args, **kwargs)

            key = f"cache:query:{func.__module__}.{func.__name__}:{make_digest(args, kwargs)}"
            try:
                cached = client.get(key)
                if cached is not None:
                    return json.loads(cached)
            except Exception as exc:
                logger.warning(f"Redis query cache read failed for {func.__name__}: {exc}")

            result = func(*args, **kwargs)
            try:
                client.setex(key, ttl_seconds, json.dumps(result, ensure_ascii=True, default=str))
            except Exception as exc:
                logger.warning(f"Redis query cache write failed for {func.__name__}: {exc}")
            return result

        return wrapper

    return decorator


def invalidate_query_cache(pattern: str = "cache:query:*") -> None:
    client = get_sync_redis_client()
    if not client:
        return

    try:
        keys = list(client.scan_iter(pattern, count=100))
        if keys:
            client.delete(*keys)
    except Exception as exc:
        logger.warning(f"Redis query cache invalidation failed: {exc}")


@contextmanager
def redis_distributed_lock(
    name: str,
    ttl_seconds: int | None = None,
    wait_timeout_seconds: float | None = None,
) -> Iterator[bool]:
    settings = get_settings()
    ttl = ttl_seconds or settings.REDIS_WRITE_LOCK_TTL_SECONDS
    wait_timeout = (
        wait_timeout_seconds
        if wait_timeout_seconds is not None
        else settings.REDIS_LOCK_WAIT_TIMEOUT_SECONDS
    )
    client = get_sync_redis_client()
    if not client:
        yield True
        return

    key = f"lock:{name}"
    token = str(uuid.uuid4())
    deadline = time.monotonic() + wait_timeout
    acquired = False
    try:
        while True:
            acquired = bool(client.set(key, token, ex=ttl, nx=True))
            if acquired or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        yield acquired
    finally:
        if acquired:
            try:
                current = client.get(key)
                if current == token:
                    client.delete(key)
            except Exception as exc:
                logger.warning(f"Redis lock release failed for {name}: {exc}")
