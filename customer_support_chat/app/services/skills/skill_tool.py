from __future__ import annotations

from langchain_core.tools import tool

from customer_support_chat.app.services.skills.loader import list_skills, load_skill


@tool
def list_available_skills() -> str:
    """
    List all available skills with short descriptions.
    """
    skills = list_skills()
    lines = []
    for s in skills:
        lines.append(
            f"- {s['name']}: {s['description']} | domain={s['domain']} | tags={','.join(s['tags'])}"
        )
    return "\n".join(lines)


@tool
def load_skill(skill_name: str) -> str:
    """
    Load a skill definition by name. Use this when a structured workflow or checklist
    would help handle the current request more reliably.
    """
    skill = load_skill(skill_name)
    meta = skill["metadata"]
    content = skill["content"]

    return (
        f"SKILL NAME: {meta['name']}\n"
        f"DOMAIN: {meta['domain']}\n"
        f"DESCRIPTION: {meta['description']}\n"
        f"RECOMMENDED AGENTS: {', '.join(meta['recommended_agents'])}\n"
        f"TAGS: {', '.join(meta['tags'])}\n\n"
        f"SKILL CONTENT:\n{content}"
    )