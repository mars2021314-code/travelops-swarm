from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ReflectionDecision(BaseModel):
    status: Literal["ok", "revise", "need_tool", "need_handoff", "need_user_input"] = Field(
        description="What the agent should do next."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the current draft / action plan."
    )
    rationale: str = Field(
        description="Short explanation of the reflection result."
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Fields or facts still missing."
    )
    suggested_target_agent: str | None = Field(
        default=None,
        description="Suggested next peer agent if handoff is recommended."
    )
    should_ask_user: bool = Field(
        default=False,
        description="Whether a user follow-up is necessary."
    )


class CriticDecision(BaseModel):
    verdict: Literal["approve", "revise"] = Field(
        description="Whether the final reply is acceptable."
    )
    rationale: str = Field(
        description="Why the answer is approved or needs revision."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific issues found in the answer."
    )
    revised_answer: str | None = Field(
        default=None,
        description="If revision is required, provide a corrected answer."
    )


class PreActionDecision(BaseModel):
    decision: Literal["approve", "ask_user", "need_tool", "revise", "handoff"] = Field(
        description="Whether the pending sensitive action may proceed."
    )
    rationale: str = Field(
        description="Why the action is approved or blocked."
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Any parameters or facts still missing."
    )
    safety_notes: list[str] = Field(
        default_factory=list,
        description="Risks, assumptions, or important cautions."
    )
    suggested_target_agent: str | None = Field(
        default=None,
        description="Suggested target if another peer agent should own the next step."
    )