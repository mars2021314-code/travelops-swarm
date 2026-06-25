from __future__ import annotations

import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.memory.experience_store import ExperienceStore

settings = get_settings()


def _tokenize(text: str) -> set[str]:
    text = text.lower()
    return set(re.findall(r"[a-z0-9_]+", text))


def _keyword_score(query: str, record: dict[str, Any]) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    haystack = " ".join(
        [
            str(record.get("memory_type", "")),
            str(record.get("summary", "")),
            str(record.get("content", "")),
            str(record.get("final_resolution", "")),
            str(record.get("entities", {})),
            " ".join(record.get("tags", [])),
        ]
    )
    record_tokens = _tokenize(haystack)
    if not record_tokens:
        return 0.0
    return len(query_tokens & record_tokens) / max(len(query_tokens), 1)


def _scope_namespaces(config: RunnableConfig | None, state: dict[str, Any]) -> dict[str, str]:
    configurable = (config or {}).get("configurable", {})
    passenger_id = configurable.get("passenger_id", "anonymous")
    thread_id = configurable.get("thread_id", "default-thread")
    agent_name = state.get("active_agent", "triage")
    return {
        "passenger_id": passenger_id,
        "thread_id": thread_id,
        "user_namespace": f"user:{passenger_id}",
        "thread_namespace": f"thread:{thread_id}",
        "agent_namespace": f"agent:{agent_name}:user:{passenger_id}",
    }


def build_experience_query(state: dict[str, Any]) -> str:
    parts = []

    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = getattr(last_msg, "content", "")
        if isinstance(content, str):
            parts.append(content)

    pending_handoff = state.get("pending_handoff", {})
    if pending_handoff:
        parts.append(str(pending_handoff.get("task_summary", "")))
        parts.append(str(pending_handoff.get("context", "")))

    working_memory = state.get("working_memory", {})
    if working_memory:
        parts.append(str(working_memory.get("memory_open_loops", [])))
        parts.append(str(working_memory.get("memory_trip_facts", [])))

    user_info = state.get("user_info", "")
    if user_info:
        parts.append(user_info)

    last_tool_result = state.get("last_tool_result", {})
    if last_tool_result:
        parts.append(str(last_tool_result))

    return "\n".join([p for p in parts if p]).strip()


def _merge_and_rank(query: str, groups: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for record in group:
            memory_id = record.get("memory_id") or record.get("memory_key") or record.get("content")
            keyword_score = _keyword_score(query, record)
            blended_score = float(record.get("score", 0.0)) + (0.2 * keyword_score)
            existing = merged.get(memory_id)
            candidate = dict(record)
            candidate["keyword_score"] = keyword_score
            candidate["blended_score"] = blended_score
            if existing is None or blended_score > existing.get("blended_score", 0.0):
                merged[memory_id] = candidate

    ranked = sorted(
        merged.values(),
        key=lambda item: item.get("blended_score", 0.0),
        reverse=True,
    )
    return ranked[:limit]


def _build_memory_context(records: list[dict[str, Any]]) -> dict[str, Any]:
    profile = []
    trip_facts = []
    open_loops = []
    candidate_options = []
    policy_snapshots = []
    episode_lessons = []

    for record in records:
        summary = record.get("summary") or record.get("final_resolution") or record.get("content", "")
        view = {
            "memory_id": record.get("memory_id"),
            "memory_type": record.get("memory_type"),
            "summary": str(summary)[:240],
            "tags": record.get("tags", []),
            "scope": record.get("scope"),
        }
        memory_type = record.get("memory_type")
        if memory_type == "user_profile":
            profile.append(view)
        elif memory_type in {"trip_fact", "session_summary", "entity_fact"}:
            trip_facts.append(view)
        elif memory_type == "open_loop":
            open_loops.append(view)
        elif memory_type == "candidate_option":
            candidate_options.append(view)
        elif memory_type == "policy_snapshot":
            policy_snapshots.append(view)
        else:
            episode_lessons.append(view)

    return {
        "memory_profile": profile[:3],
        "memory_trip_facts": trip_facts[:5],
        "memory_open_loops": open_loops[:5],
        "memory_candidate_options": candidate_options[:8],
        "memory_policy_snapshots": policy_snapshots[:3],
        "memory_episode_lessons": episode_lessons[:5],
    }


def _brief_lines(items: list[dict[str, Any]], label: str, limit: int) -> list[str]:
    lines = []
    for item in items[:limit]:
        summary = str(item.get("summary", "")).strip()
        if not summary:
            continue
        lines.append(f"{label}: {summary}")
    return lines


def _build_agent_brief(memory_context: dict[str, Any]) -> str:
    lines = []
    lines.extend(_brief_lines(memory_context.get("memory_profile", []), "Profile", 2))
    lines.extend(_brief_lines(memory_context.get("memory_trip_facts", []), "Fact", 4))
    lines.extend(_brief_lines(memory_context.get("memory_candidate_options", []), "Option", 4))
    lines.extend(_brief_lines(memory_context.get("memory_policy_snapshots", []), "Policy", 2))
    lines.extend(_brief_lines(memory_context.get("memory_open_loops", []), "Open loop", 3))
    lines.extend(_brief_lines(memory_context.get("memory_episode_lessons", []), "Lesson", 2))
    return "\n".join(lines[:14]).strip()


def retrieve_experiences(
    state: dict[str, Any],
    store: ExperienceStore,
    config: RunnableConfig | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    query = build_experience_query(state)
    scope_meta = _scope_namespaces(config, state)
    limit = top_k or settings.MEMORY_TOP_K

    groups = [
        store.search(
            query,
            top_k=3,
            scope="user",
            namespace=scope_meta["user_namespace"],
            memory_types=["user_profile"],
            passenger_id=scope_meta["passenger_id"],
        ),
        store.search(
            query,
            top_k=6,
            scope="thread",
            namespace=scope_meta["thread_namespace"],
            memory_types=[
                "trip_fact",
                "open_loop",
                "session_summary",
                "entity_fact",
                "candidate_option",
                "policy_snapshot",
            ],
            passenger_id=scope_meta["passenger_id"],
            thread_id=scope_meta["thread_id"],
        ),
        store.search(
            query,
            top_k=2,
            scope="agent",
            namespace=scope_meta["agent_namespace"],
            passenger_id=scope_meta["passenger_id"],
            thread_id=scope_meta["thread_id"],
            agent_name=state.get("active_agent", "triage"),
        ),
        store.search(
            query,
            top_k=max(limit, 4),
            scope="episode",
            namespace=scope_meta["user_namespace"],
            memory_types=["episode_resolution"],
            passenger_id=scope_meta["passenger_id"],
        ),
    ]

    ranked = _merge_and_rank(query, groups, limit)
    memory_context = _build_memory_context(ranked)
    agent_brief = _build_agent_brief(memory_context)
    working_memory = {
        "memory_query": query,
        "retrieved_experience_count": len(ranked),
        "agent_brief": agent_brief,
        **memory_context,
    }

    return {
        "hits": ranked,
        "memory_context": memory_context,
        "working_memory": working_memory,
    }
