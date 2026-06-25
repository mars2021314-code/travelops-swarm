from __future__ import annotations

from customer_support_chat.app.services.skills.registry import SKILL_REGISTRY
from customer_support_chat.app.services.skills.skill_store import read_skill_markdown


def list_skills() -> list[dict]:
    return list(SKILL_REGISTRY.values())


def get_skill_metadata(skill_name: str) -> dict:
    if skill_name not in SKILL_REGISTRY:
        raise ValueError(f"Unknown skill: {skill_name}")
    return SKILL_REGISTRY[skill_name]


def load_skill(skill_name: str) -> dict:
    metadata = get_skill_metadata(skill_name)
    content = read_skill_markdown(skill_name)
    return {
        "metadata": metadata,
        "content": content,
    }