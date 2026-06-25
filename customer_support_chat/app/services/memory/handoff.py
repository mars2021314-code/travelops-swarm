from __future__ import annotations

from customer_support_chat.app.core.state import State


def memory_target_keywords(target: str) -> set[str]:
    keyword_map = {
        "triage": {"policy_snapshot", "faq_option", "flight_option"},
        "update_flight": {"ticket", "flight", "flight_option", "policy_snapshot"},
        "book_hotel": {"hotel", "hotel_option", "policy_snapshot"},
        "book_car_rental": {"car_rental", "car_rental_option", "policy_snapshot"},
        "book_excursion": {"excursion", "excursion_option", "policy_snapshot"},
    }
    return keyword_map.get(target, set())


def is_relevant_memory_item(item: dict, target: str) -> bool:
    target_keywords = memory_target_keywords(target)
    item_type = str(item.get("memory_type", ""))
    tags = {str(tag) for tag in item.get("tags", [])}
    summary = str(item.get("summary", "")).lower()

    if not target_keywords:
        return True

    if item_type in {"open_loop", "session_summary", "trip_fact"}:
        return True

    for keyword in target_keywords:
        if keyword in item_type or keyword in summary or keyword in tags:
            return True
    return False


def compact_memory_items(items: list[dict], target: str, limit: int) -> list[str]:
    lines = []
    for item in items:
        if not is_relevant_memory_item(item, target):
            continue
        summary = str(item.get("summary", "")).strip()
        if not summary:
            continue
        memory_type = str(item.get("memory_type", "memory"))
        lines.append(f"- [{memory_type}] {summary}")
        if len(lines) >= limit:
            break
    return lines


def build_handoff_memory_brief(state: State, target: str) -> str:
    working_memory = state.get("working_memory", {}) or {}
    sections = []

    current_facts = compact_memory_items(
        working_memory.get("memory_trip_facts", []) or [],
        target,
        limit=4,
    )
    if current_facts:
        sections.append("Current confirmed facts:\n" + "\n".join(current_facts))

    candidate_options = compact_memory_items(
        working_memory.get("memory_candidate_options", []) or [],
        target,
        limit=4,
    )
    if candidate_options:
        sections.append("Relevant candidate options:\n" + "\n".join(candidate_options))

    policy_snapshots = compact_memory_items(
        working_memory.get("memory_policy_snapshots", []) or [],
        target,
        limit=2,
    )
    if policy_snapshots:
        sections.append("Recent policy evidence:\n" + "\n".join(policy_snapshots))

    open_loops = compact_memory_items(
        working_memory.get("memory_open_loops", []) or [],
        target,
        limit=4,
    )
    if open_loops:
        sections.append("Outstanding open loops:\n" + "\n".join(open_loops))

    return "\n\n".join(sections).strip()


def augment_handoff_payload(state: State, target: str, args: dict) -> tuple[str, str]:
    task_summary = str(args.get("task_summary", "")).strip()
    context = str(args.get("context", "")).strip()
    memory_brief = build_handoff_memory_brief(state, target)

    if not task_summary:
        task_summary = f"Continue assisting with the {target} workflow."

    if memory_brief:
        context = "\n\n".join(part for part in [context, memory_brief] if part)

    return task_summary, context
