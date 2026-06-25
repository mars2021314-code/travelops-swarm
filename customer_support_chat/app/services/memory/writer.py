from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig

from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.memory.experience_store import ExperienceStore

settings = get_settings()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_in_hours(hours: int | None) -> str | None:
    if hours is None or hours <= 0:
        return None
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _guess_issue_type(state: dict[str, Any]) -> str:
    active_agent = state.get("active_agent", "triage")
    mapping = {
        "triage": "general_support",
        "update_flight": "flight_change_or_cancellation",
        "book_car_rental": "car_rental",
        "book_hotel": "hotel_booking",
        "book_excursion": "excursion_booking",
    }
    return mapping.get(active_agent, "general_support")


def _extract_latest_user_message(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", "")
        if msg_type == "human":
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content
    return ""


def _extract_open_loops(state: dict[str, Any]) -> list[str]:
    loops = []
    pending_handoff = state.get("pending_handoff", {})
    if pending_handoff:
        task_summary = str(pending_handoff.get("task_summary", "")).strip()
        if task_summary:
            loops.append(task_summary)

    reflection_trace = state.get("reflection_trace", [])
    if reflection_trace:
        last_result = reflection_trace[-1].get("result", {})
        for item in last_result.get("missing_information", []):
            text = str(item).strip()
            if text:
                loops.append(text)
    return loops


def _config_scope(config: RunnableConfig | None, state: dict[str, Any]) -> dict[str, str]:
    configurable = (config or {}).get("configurable", {})
    passenger_id = configurable.get("passenger_id", "anonymous")
    thread_id = configurable.get("thread_id", "default-thread")
    agent_name = state.get("active_agent", "triage")
    return {
        "passenger_id": passenger_id,
        "thread_id": thread_id,
        "agent_name": agent_name,
        "user_namespace": f"user:{passenger_id}",
        "thread_namespace": f"thread:{thread_id}",
        "agent_namespace": f"agent:{agent_name}:user:{passenger_id}",
    }


def _build_profile_memory(state: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    return {
        "memory_key": f"profile:{scope['passenger_id']}",
        "memory_type": "user_profile",
        "scope": "user",
        "namespace": scope["user_namespace"],
        "passenger_id": scope["passenger_id"],
        "thread_id": scope["thread_id"],
        "agent_name": scope["agent_name"],
        "summary": "Current customer profile and booked flight context.",
        "content": state.get("user_info", ""),
        "entities": {
            "user_info": state.get("user_info", ""),
        },
        "tags": ["profile", "flight_context"],
        "valid_from": _utc_now(),
        "valid_to": None,
        "is_active": True,
        "updated_at": _utc_now(),
    }


def _build_trip_fact_memories(state: dict[str, Any], scope: dict[str, str]) -> list[dict[str, Any]]:
    latest_user_msg = _extract_latest_user_message(state)
    memories = [
        {
            "memory_key": f"trip-state:{scope['thread_id']}",
            "memory_type": "trip_fact",
            "scope": "thread",
            "namespace": scope["thread_namespace"],
            "passenger_id": scope["passenger_id"],
            "thread_id": scope["thread_id"],
            "agent_name": scope["agent_name"],
            "summary": "Latest user request and current trip state.",
            "content": latest_user_msg,
            "entities": {
                "latest_user_message": latest_user_msg,
                "user_info": state.get("user_info", ""),
            },
            "tags": ["trip_state", scope["agent_name"]],
            "valid_from": _utc_now(),
            "valid_to": None,
            "is_active": True,
            "updated_at": _utc_now(),
        }
    ]

    for loop in _extract_open_loops(state):
        memories.append(
            {
                "memory_key": f"open-loop:{scope['thread_id']}:{loop[:80]}",
                "memory_type": "open_loop",
                "scope": "thread",
                "namespace": scope["thread_namespace"],
                "passenger_id": scope["passenger_id"],
                "thread_id": scope["thread_id"],
                "agent_name": scope["agent_name"],
                "summary": loop[:240],
                "content": loop,
                "entities": {
                    "active_agent": scope["agent_name"],
                },
                "tags": ["open_loop", scope["agent_name"]],
                "valid_from": _utc_now(),
                "valid_to": None,
                "expires_at": _expires_in_hours(settings.MEMORY_OPEN_LOOP_TTL_HOURS),
                "is_active": True,
                "updated_at": _utc_now(),
            }
        )

    return memories


def _build_handoff_memory(state: dict[str, Any], scope: dict[str, str]) -> list[dict[str, Any]]:
    pending_handoff = state.get("pending_handoff", {}) or {}
    if not pending_handoff:
        return []

    summary = str(pending_handoff.get("task_summary", "")).strip()
    context = str(pending_handoff.get("context", "")).strip()
    if not summary and not context:
        return []

    content = "\n".join(part for part in [summary, context] if part)
    target_agent = pending_handoff.get("to_agent", "triage")
    return [
        {
            "memory_key": f"handoff:{scope['thread_id']}:{target_agent}:{summary[:80]}",
            "memory_type": "handoff_brief",
            "scope": "thread",
            "namespace": scope["thread_namespace"],
            "passenger_id": scope["passenger_id"],
            "thread_id": scope["thread_id"],
            "agent_name": scope["agent_name"],
            "summary": summary[:240] or f"Handoff to {target_agent}",
            "content": content,
            "entities": pending_handoff,
            "tags": ["handoff", target_agent, scope["agent_name"]],
            "valid_from": _utc_now(),
            "valid_to": None,
            "expires_at": _expires_in_hours(settings.MEMORY_HANDOFF_TTL_HOURS),
            "is_active": True,
            "updated_at": _utc_now(),
        }
    ]


def _build_tool_observation_memories(state: dict[str, Any], scope: dict[str, str]) -> list[dict[str, Any]]:
    last_tool_result = state.get("last_tool_result", {}) or {}
    tool_outputs = last_tool_result.get("tool_outputs", []) or []
    memories = []

    for index, tool_output in enumerate(tool_outputs):
        content = str(tool_output.get("content", "")).strip()
        if not content:
            continue
        tool_name = str(tool_output.get("tool_name", "")).strip() or "tool"
        memories.append(
            {
                "memory_key": f"tool-observation:{scope['thread_id']}:{tool_name}:{content[:80]}",
                "memory_type": "tool_observation",
                "scope": "agent",
                "namespace": scope["agent_namespace"],
                "passenger_id": scope["passenger_id"],
                "thread_id": scope["thread_id"],
                "agent_name": scope["agent_name"],
                "summary": f"{tool_name}: {content[:180]}",
                "content": content,
                "entities": {
                    "tool_name": tool_name,
                    "tool_scope": last_tool_result.get("tool_scope"),
                    "tool_call_id": tool_output.get("tool_call_id", ""),
                },
                "tags": ["tool_observation", tool_name, scope["agent_name"]],
                "valid_from": _utc_now(),
                "valid_to": None,
                "expires_at": _expires_in_hours(settings.MEMORY_TOOL_OBSERVATION_TTL_HOURS),
                "is_active": True,
                "updated_at": _utc_now(),
            }
        )

    return memories


def _parse_tool_payload(content: str) -> Any:
    text = content.strip()
    if not text:
        return None

    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return text


def _build_candidate_snapshot(
    *,
    scope: dict[str, str],
    tool_name: str,
    item: dict[str, Any],
    entity_type: str,
    entity_id_key: str,
    summary_fields: list[str],
) -> dict[str, Any] | None:
    entity_id = item.get(entity_id_key)
    if entity_id is None:
        return None

    summary_parts = []
    for field in summary_fields:
        value = item.get(field)
        if value not in (None, "", []):
            summary_parts.append(f"{field}={value}")
    summary = ", ".join(summary_parts)[:240] or f"{entity_type} candidate {entity_id}"

    return {
        "memory_key": f"candidate:{scope['thread_id']}:{tool_name}:{entity_type}:{entity_id}",
        "memory_type": "candidate_option",
        "scope": "thread",
        "namespace": scope["thread_namespace"],
        "passenger_id": scope["passenger_id"],
        "thread_id": scope["thread_id"],
        "agent_name": scope["agent_name"],
        "summary": summary,
        "content": json.dumps(item, ensure_ascii=False),
        "entity_type": entity_type,
        "entity_id": str(entity_id),
        "entities": item,
        "tags": ["candidate_option", entity_type, tool_name],
        "valid_from": _utc_now(),
        "valid_to": None,
        "expires_at": _expires_in_hours(settings.MEMORY_CANDIDATE_TTL_HOURS),
        "is_active": True,
        "updated_at": _utc_now(),
        "deactivation_filters": [
            {
                "memory_type": "candidate_option",
                "scope": "thread",
                "namespace": scope["thread_namespace"],
                "entity_type": entity_type,
                "entity_id": str(entity_id),
                "is_active": True,
            }
        ],
    }


def _build_tool_snapshot_memories(state: dict[str, Any], scope: dict[str, str]) -> list[dict[str, Any]]:
    last_tool_result = state.get("last_tool_result", {}) or {}
    tool_outputs = last_tool_result.get("tool_outputs", []) or []
    memories = []

    for tool_output in tool_outputs:
        tool_name = str(tool_output.get("tool_name", "")).strip()
        content = str(tool_output.get("content", "")).strip()
        if not tool_name or not content:
            continue

        parsed = _parse_tool_payload(content)

        if tool_name in {
            "search_flights",
            "search_hotels",
            "search_car_rentals",
            "search_trip_recommendations",
            "search_faq",
        } and isinstance(parsed, list):
            schema_map = {
                "search_flights": ("flight_option", "flight_id", ["flight_no", "departure_airport", "arrival_airport", "scheduled_departure"]),
                "search_hotels": ("hotel_option", "id", ["name", "location", "price_tier", "checkin_date", "checkout_date"]),
                "search_car_rentals": ("car_rental_option", "id", ["name", "location", "price_tier", "start_date", "end_date"]),
                "search_trip_recommendations": ("excursion_option", "id", ["name", "location", "keywords"]),
                "search_faq": ("faq_option", "question", ["question", "category", "similarity"]),
            }
            entity_type, entity_id_key, summary_fields = schema_map[tool_name]
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                memory = _build_candidate_snapshot(
                    scope=scope,
                    tool_name=tool_name,
                    item=item,
                    entity_type=entity_type,
                    entity_id_key=entity_id_key,
                    summary_fields=summary_fields,
                )
                if memory is not None:
                    memories.append(memory)
            continue

        if tool_name == "lookup_policy":
            memories.append(
                {
                    "memory_key": f"policy-snapshot:{scope['thread_id']}:{content[:80]}",
                    "memory_type": "policy_snapshot",
                    "scope": "thread",
                    "namespace": scope["thread_namespace"],
                    "passenger_id": scope["passenger_id"],
                    "thread_id": scope["thread_id"],
                    "agent_name": scope["agent_name"],
                    "summary": content[:240],
                    "content": content[:2000],
                    "entities": {
                        "tool_name": tool_name,
                    },
                    "tags": ["policy_snapshot", scope["agent_name"]],
                    "valid_from": _utc_now(),
                    "valid_to": None,
                    "expires_at": _expires_in_hours(settings.MEMORY_POLICY_TTL_HOURS),
                    "is_active": True,
                    "updated_at": _utc_now(),
                    "deactivation_filters": [
                        {
                            "memory_type": "policy_snapshot",
                            "scope": "thread",
                            "namespace": scope["thread_namespace"],
                            "is_active": True,
                        }
                    ],
                }
            )

    return memories


def _match(pattern: str, text: str) -> re.Match[str] | None:
    return re.search(pattern, text, flags=re.IGNORECASE)


def _extract_entity_fact(tool_name: str, content: str) -> dict[str, Any] | None:
    text = content.strip()

    if tool_name == "update_ticket_to_new_flight":
        match = _match(r"Ticket\s+(.+?)\s+successfully updated to flight\s+(\d+)", text)
        if match:
            return {
                "entity_type": "ticket",
                "entity_id": match.group(1).strip(),
                "status": "updated",
                "summary": text,
                "entities": {
                    "ticket_no": match.group(1).strip(),
                    "flight_id": int(match.group(2)),
                },
            }

    if tool_name == "cancel_ticket":
        match = _match(r"Ticket\s+(.+?)\s+successfully cancelled", text)
        if match:
            return {
                "entity_type": "ticket",
                "entity_id": match.group(1).strip(),
                "status": "cancelled",
                "summary": text,
                "entities": {
                    "ticket_no": match.group(1).strip(),
                },
            }

    simple_patterns = {
        "book_hotel": ("hotel", "booked", r"Hotel\s+(\d+)\s+successfully booked"),
        "update_hotel": ("hotel", "updated", r"Hotel\s+(\d+)\s+successfully updated"),
        "cancel_hotel": ("hotel", "cancelled", r"Hotel\s+(\d+)\s+successfully cancelled"),
        "book_car_rental": ("car_rental", "booked", r"Car rental\s+(\d+)\s+successfully booked"),
        "update_car_rental": ("car_rental", "updated", r"Car rental\s+(\d+)\s+successfully updated"),
        "cancel_car_rental": ("car_rental", "cancelled", r"Car rental\s+(\d+)\s+successfully cancelled"),
        "book_excursion": ("excursion", "booked", r"Excursion\s+(\d+)\s+successfully booked"),
        "update_excursion": ("excursion", "updated", r"Excursion\s+(\d+)\s+successfully updated"),
        "cancel_excursion": ("excursion", "cancelled", r"Excursion\s+(\d+)\s+successfully cancelled"),
    }

    if tool_name in simple_patterns:
        entity_type, status, pattern = simple_patterns[tool_name]
        match = _match(pattern, text)
        if match:
            return {
                "entity_type": entity_type,
                "entity_id": match.group(1),
                "status": status,
                "summary": text,
                "entities": {
                    "id": int(match.group(1)),
                },
            }

    return None


def _build_entity_fact_memories(state: dict[str, Any], scope: dict[str, str]) -> list[dict[str, Any]]:
    last_tool_result = state.get("last_tool_result", {}) or {}
    tool_outputs = last_tool_result.get("tool_outputs", []) or []
    memories = []

    for index, tool_output in enumerate(tool_outputs):
        content = str(tool_output.get("content", "")).strip()
        if not content:
            continue
        tool_name = str(tool_output.get("tool_name", "")).strip() or "tool"
        fact = _extract_entity_fact(tool_name, content)
        if fact is None:
            continue

        entity_key = f"{fact['entity_type']}:{fact['entity_id']}"
        memories.append(
            {
                "memory_key": f"entity-fact:{scope['thread_id']}:{entity_key}",
                "memory_type": "entity_fact",
                "scope": "thread",
                "namespace": scope["thread_namespace"],
                "passenger_id": scope["passenger_id"],
                "thread_id": scope["thread_id"],
                "agent_name": scope["agent_name"],
                "summary": fact["summary"][:240],
                "content": content,
                "entity_type": fact["entity_type"],
                "entity_id": str(fact["entity_id"]),
                "status": fact["status"],
                "entities": fact["entities"],
                "tags": ["entity_fact", fact["entity_type"], fact["status"]],
                "valid_from": _utc_now(),
                "valid_to": None,
                "is_active": True,
                "updated_at": _utc_now(),
                "deactivation_filters": [
                    {
                        "memory_type": "entity_fact",
                        "scope": "thread",
                        "namespace": scope["thread_namespace"],
                        "entity_type": fact["entity_type"],
                        "entity_id": str(fact["entity_id"]),
                        "is_active": True,
                    }
                ],
            }
        )

    return memories


def _build_session_summary_memory(state: dict[str, Any], scope: dict[str, str]) -> list[dict[str, Any]]:
    final_resolution = ""
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "ai":
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                final_resolution = content.strip()
                break

    if not final_resolution:
        return []

    return [
        {
            "memory_key": f"session-summary:{scope['thread_id']}",
            "memory_type": "session_summary",
            "scope": "thread",
            "namespace": scope["thread_namespace"],
            "passenger_id": scope["passenger_id"],
            "thread_id": scope["thread_id"],
            "agent_name": scope["agent_name"],
            "summary": final_resolution[:240],
            "content": final_resolution[:1000],
            "entities": {
                "issue_type": _guess_issue_type(state),
            },
            "tags": ["session_summary", scope["agent_name"]],
            "valid_from": _utc_now(),
            "valid_to": None,
            "expires_at": _expires_in_hours(settings.MEMORY_SESSION_SUMMARY_TTL_HOURS),
            "is_active": True,
            "updated_at": _utc_now(),
            "deactivation_filters": [
                {
                    "memory_type": "session_summary",
                    "scope": "thread",
                    "namespace": scope["thread_namespace"],
                    "is_active": True,
                }
            ],
        }
    ]


def build_experience_record(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    latest_user_msg = _extract_latest_user_message(state)
    handoff_history = state.get("handoff_history", [])
    reflection_trace = state.get("reflection_trace", [])
    last_tool_result = state.get("last_tool_result", {})
    working_memory = state.get("working_memory", {})
    experience_hits = state.get("experience_hits", [])
    scope = _config_scope(config, state)

    tool_sequence = []
    if last_tool_result:
        pending = last_tool_result.get("pending_sensitive_action", {}) or {}
        tool_sequence.extend(pending.get("tool_names", []))

    final_resolution = ""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", "")
        if msg_type == "ai":
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                final_resolution = content.strip()
                break

    return {
        "memory_type": "episode_resolution",
        "scope": "episode",
        "namespace": scope["user_namespace"],
        "passenger_id": scope["passenger_id"],
        "thread_id": scope["thread_id"],
        "agent_name": scope["agent_name"],
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "valid_from": _utc_now(),
        "valid_to": None,
        "is_active": True,
        "issue_type": _guess_issue_type(state),
        "intent_signature": latest_user_msg[:500],
        "summary": latest_user_msg[:240] or "Customer support interaction episode.",
        "entities": {
            "working_memory": working_memory,
            "memory_context": state.get("memory_context", {}),
            "pending_handoff": state.get("pending_handoff", {}),
        },
        "tool_sequence": tool_sequence,
        "handoff_history": handoff_history,
        "reflection_summary": reflection_trace[-3:],
        "final_resolution": final_resolution[:1000],
        "outcome": "success" if final_resolution else "partial",
        "failure_reason": None if final_resolution else "No final AI resolution captured.",
        "tags": [state.get("active_agent", "triage"), "episode"],
        "supersedes": [
            hit.get("memory_id")
            for hit in experience_hits
            if isinstance(hit, dict)
            and hit.get("memory_type") in {"episode_resolution", "session_summary", "trip_fact"}
        ],
        "source_experience_ids": [
            hit.get("memory_id")
            for hit in experience_hits
            if isinstance(hit, dict) and hit.get("memory_id")
        ],
    }


def write_state_memories(
    state: dict[str, Any],
    store: ExperienceStore,
    config: RunnableConfig | None = None,
) -> list[dict[str, Any]]:
    scope = _config_scope(config, state)
    records = [_build_profile_memory(state, scope)]
    records.extend(_build_trip_fact_memories(state, scope))
    records.extend(_build_handoff_memory(state, scope))
    records.extend(_build_tool_observation_memories(state, scope))
    records.extend(_build_tool_snapshot_memories(state, scope))
    records.extend(_build_entity_fact_memories(state, scope))
    records.extend(_build_session_summary_memory(state, scope))
    stored = []
    for record in records:
        stored.append(store.append(record, write_audit=False))
    return stored


def write_experience(
    state: dict[str, Any],
    store: ExperienceStore,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    record = build_experience_record(state, config=config)
    return store.append(record, write_audit=True)
