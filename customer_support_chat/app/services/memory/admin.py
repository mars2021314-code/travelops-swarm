from __future__ import annotations

from typing import Any

from customer_support_chat.app.services.memory.experience_store import ExperienceStore


def scope_filters(
    *,
    scope: str,
    passenger_id: str,
    thread_id: str,
) -> dict[str, Any]:
    if scope == "thread":
        return {
            "thread_id": thread_id,
            "scope": ["thread", "agent", "episode"],
        }
    if scope == "user":
        return {
            "passenger_id": passenger_id,
        }
    return {}


def inspect_memory(
    *,
    store: ExperienceStore,
    passenger_id: str,
    thread_id: str,
    scope: str = "thread",
    include_inactive: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    filters = scope_filters(scope=scope, passenger_id=passenger_id, thread_id=thread_id)
    summary = store.summarize_memories(
        filters=filters,
        include_inactive=include_inactive,
        limit=limit,
    )
    memories = store.list_memories(
        filters=filters,
        include_inactive=include_inactive,
        limit=limit,
    )
    return {
        "scope": scope,
        "summary": summary,
        "memories": memories,
    }


def forget_memory(
    *,
    store: ExperienceStore,
    passenger_id: str,
    thread_id: str,
    scope: str = "thread",
) -> dict[str, Any]:
    filters = scope_filters(scope=scope, passenger_id=passenger_id, thread_id=thread_id)
    purged = store.purge_memories(filters=filters, include_inactive=True)
    return {
        "scope": scope,
        "purged_count": purged,
    }


def expire_memory(
    *,
    store: ExperienceStore,
) -> dict[str, Any]:
    expired = store.expire_stale_memories()
    return {
        "expired_count": expired,
    }
