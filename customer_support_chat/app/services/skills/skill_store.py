from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SKILL_DEFINITIONS_DIR = BASE_DIR / "definitions"


def get_skill_path(skill_name: str) -> Path:
    return SKILL_DEFINITIONS_DIR / skill_name / "SKILL.md"


def read_skill_markdown(skill_name: str) -> str:
    path = get_skill_path(skill_name)
    if not path.exists():
        raise FileNotFoundError(f"Skill '{skill_name}' not found at {path}")
    return path.read_text(encoding="utf-8")