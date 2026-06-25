from __future__ import annotations

import json
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from customer_support_chat.app.services.memory.admin import (
    expire_memory,
    forget_memory,
    inspect_memory,
)
from customer_support_chat.app.services.memory.experience_store import ExperienceStore

memory_store = ExperienceStore()


def _config_ids(config: RunnableConfig) -> tuple[str, str]:
    configurable = config.get("configurable", {})
    passenger_id = configurable.get("passenger_id", "anonymous")
    thread_id = configurable.get("thread_id", "default-thread")
    return passenger_id, thread_id


@tool
def inspect_memory_state(
    scope: Literal["thread", "user", "all"] = "thread",
    include_inactive: bool = False,
    limit: int = 20,
    *,
    config: RunnableConfig,
) -> str:
    """Inspect stored memory for debugging, transparency, or troubleshooting."""
    passenger_id, thread_id = _config_ids(config)
    result = inspect_memory(
        store=memory_store,
        passenger_id=passenger_id,
        thread_id=thread_id,
        scope=scope,
        include_inactive=include_inactive,
        limit=limit,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def forget_memory_scope(
    scope: Literal["thread", "user", "all"] = "thread",
    *,
    config: RunnableConfig,
) -> str:
    """Delete stored memory for the current thread or user. Use only when the user explicitly asks to forget memory."""
    passenger_id, thread_id = _config_ids(config)
    result = forget_memory(
        store=memory_store,
        passenger_id=passenger_id,
        thread_id=thread_id,
        scope=scope,
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def expire_stale_memory() -> str:
    """Expire stale memories based on TTL metadata."""
    result = expire_memory(store=memory_store)
    return json.dumps(result, ensure_ascii=False)
