from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


def set_or_keep(left, right):
    return left if right is None else right


def append_list(left: Optional[list], right: Optional[list]) -> list:
    left = left or []
    right = right or []
    return left + right


def merge_dict(left: Optional[dict], right: Optional[dict]) -> dict:
    left = left or {}
    right = right or {}
    merged = dict(left)
    merged.update(right)
    return merged


class HandoffRecord(TypedDict, total=False):
    from_agent: str
    to_agent: str
    reason: str
    tool_name: str
    tool_call_id: str


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str

    active_agent: Annotated[str, set_or_keep]
    last_active_agent: Annotated[str, set_or_keep]
    swarm_stack: Annotated[list[str], append_list]

    task_board: Annotated[list[dict[str, Any]], append_list]
    handoff_history: Annotated[list[HandoffRecord], append_list]
    pending_tool_scope: Annotated[str, set_or_keep]
    pending_handoff: Annotated[dict[str, Any], merge_dict]
    pending_sensitive_action: Annotated[dict[str, Any], merge_dict]

    working_memory: Annotated[dict[str, Any], merge_dict]
    memory_context: Annotated[dict[str, Any], merge_dict]
    reflection_trace: Annotated[list[dict[str, Any]], append_list]
    experience_hits: Annotated[list[dict[str, Any]], append_list]
    experience_candidate: Annotated[dict[str, Any], merge_dict]

    approval_required: Annotated[bool, set_or_keep]
    last_tool_result: Annotated[dict[str, Any], merge_dict]
