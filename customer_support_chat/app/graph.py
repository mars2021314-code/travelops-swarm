from __future__ import annotations
from typing import Literal, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from customer_support_chat.app.core.state import State
from customer_support_chat.app.core.checkpoint import build_checkpointer
from customer_support_chat.app.services.assistants.assistant_base import (
    CompleteOrEscalate,
)
from customer_support_chat.app.services.assistants.car_rental_assistant import (
    book_car_rental_safe_tools,
    book_car_rental_sensitive_tools,
    car_rental_assistant,
)
from customer_support_chat.app.services.assistants.excursion_assistant import (
    book_excursion_safe_tools,
    book_excursion_sensitive_tools,
    excursion_assistant,
)
from customer_support_chat.app.services.assistants.flight_booking_assistant import (
    flight_booking_assistant,
    update_flight_safe_tools,
    update_flight_sensitive_tools,
)
from customer_support_chat.app.services.assistants.hotel_booking_assistant import (
    book_hotel_safe_tools,
    book_hotel_sensitive_tools,
    hotel_booking_assistant,
)
from customer_support_chat.app.services.assistants.primary_assistant import (
    primary_assistant,
    primary_assistant_tools,
)
from customer_support_chat.app.services.memory.experience_store import ExperienceStore
from customer_support_chat.app.services.memory.handoff import (
    augment_handoff_payload,
    build_handoff_memory_brief,
)
from customer_support_chat.app.services.memory.retriever import retrieve_experiences
from customer_support_chat.app.services.memory.writer import (
    write_experience,
    write_state_memories,
)
from customer_support_chat.app.services.reflection.critic import critic_chain
from customer_support_chat.app.services.reflection.pre_action import pre_action_chain
from customer_support_chat.app.services.reflection.self_reflect import self_reflection_chain
from customer_support_chat.app.services.tools.flights import fetch_user_flight_information
from customer_support_chat.app.services.utils import (
    create_tool_node_with_fallback,
    flight_info_to_string,
)

AGENT_REGISTRY = {
    "triage": {
        "assistant": primary_assistant,
        "safe_tools": primary_assistant_tools,
        "sensitive_tools": [],
    },
    "update_flight": {
        "assistant": flight_booking_assistant,
        "safe_tools": update_flight_safe_tools,
        "sensitive_tools": update_flight_sensitive_tools,
    },
    "book_car_rental": {
        "assistant": car_rental_assistant,
        "safe_tools": book_car_rental_safe_tools,
        "sensitive_tools": book_car_rental_sensitive_tools,
    },
    "book_hotel": {
        "assistant": hotel_booking_assistant,
        "safe_tools": book_hotel_safe_tools,
        "sensitive_tools": book_hotel_sensitive_tools,
    },
    "book_excursion": {
        "assistant": excursion_assistant,
        "safe_tools": book_excursion_safe_tools,
        "sensitive_tools": book_excursion_sensitive_tools,
    },
}

experience_store = ExperienceStore()


def user_info(state: State, config: RunnableConfig):
    flight_info = fetch_user_flight_information.invoke(input={}, config=config)
    user_info_str = flight_info_to_string(flight_info)
    return {"user_info": user_info_str}


def bootstrap_swarm(state: State) -> dict:
    return {
        "active_agent": "triage",
        "last_active_agent": "triage",
        "pending_tool_scope": None,
        "pending_handoff": {},
        "pending_sensitive_action": {},
        "approval_required": False,
        "working_memory": {},
        "memory_context": {},
        "reflection_trace": [],
        "last_tool_result": {},
        "experience_hits": [],
        "experience_candidate": {},
    }


def capture_state_memory(state: State, config: RunnableConfig) -> dict:
    records = write_state_memories(state, experience_store, config=config)
    return {
        "memory_context": {
            "state_memory_write_count": len(records),
        }
    }


def refresh_state_memory(state: State, config: RunnableConfig) -> dict:
    return capture_state_memory(state, config)


def retrieve_experience_memory(state: State, config: RunnableConfig) -> dict:
    retrieval = retrieve_experiences(state, experience_store, config=config, top_k=8)
    hits = retrieval["hits"]
    return {
        "experience_hits": hits,
        "memory_context": retrieval["memory_context"],
        "working_memory": retrieval["working_memory"],
    }


def write_experience_memory(state: State, config: RunnableConfig) -> dict:
    record = write_experience(state, experience_store, config=config)
    return {
        "experience_candidate": record
    }


def _get_last_message(state: State):
    return state["messages"][-1]


def _get_last_tool_calls(state: State) -> list[dict]:
    msg = _get_last_message(state)
    return getattr(msg, "tool_calls", []) or []


def _tool_name(tool) -> str:
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    if not name:
        raise ValueError(f"Tool {tool!r} does not expose a name")
    return name


def _safe_tool_names(agent_name: str) -> set[str]:
    return {_tool_name(tool) for tool in AGENT_REGISTRY[agent_name]["safe_tools"]}


def _sensitive_tool_names(agent_name: str) -> set[str]:
    return {_tool_name(tool) for tool in AGENT_REGISTRY[agent_name]["sensitive_tools"]}


def _parse_handoff_tool(tool_name: str, args: dict) -> Optional[str]:
    if tool_name == "HandoffToAgent":
        target = args.get("target_agent")
        if target in AGENT_REGISTRY:
            return target

    legacy_mapping = {
        "ToFlightBookingAssistant": "update_flight",
        "ToBookCarRental": "book_car_rental",
        "ToHotelBookingAssistant": "book_hotel",
        "ToBookExcursion": "book_excursion",
    }
    return legacy_mapping.get(tool_name)


def _is_pending_handoff_for_agent(state: State, agent_name: str) -> bool:
    pending = state.get("pending_handoff") or {}
    if not pending:
        return False
    return pending.get("to_agent") == agent_name and not pending.get("consumed", False)


def _extract_sensitive_action(state: State, agent_name: str) -> dict:
    tool_calls = _get_last_tool_calls(state)
    sensitive_names = _sensitive_tool_names(agent_name)

    selected = []
    for tc in tool_calls:
        if tc["name"] in sensitive_names:
            selected.append(tc)

    return {
        "agent": agent_name,
        "tool_calls": selected,
        "tool_names": [tc["name"] for tc in selected],
    }


def route_dispatch_active_agent(
    state: State,
) -> Literal["triage", "update_flight", "book_car_rental", "book_hotel", "book_excursion"]:
    active = state.get("active_agent", "triage")
    if active not in AGENT_REGISTRY:
        return "triage"
    return active  # type: ignore[return-value]


def make_agent_router(agent_name: str):
    safe_names = _safe_tool_names(agent_name)
    sensitive_names = _sensitive_tool_names(agent_name)

    def router(
        state: State,
    ) -> Literal["handle_handoff", "self_reflect", "pre_action_check", "dispatch_active_agent", "__end__", str]:
        route = tools_condition(state)
        if route == END:
            return "self_reflect"

        tool_calls = _get_last_tool_calls(state)
        if not tool_calls:
            return "self_reflect"

        if any(_parse_handoff_tool(tc["name"], tc.get("args", {}) or {}) is not None for tc in tool_calls):
            return "handle_handoff"

        if any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls):
            return "handle_handoff"

        tool_names = [tc["name"] for tc in tool_calls]

        if all(name in safe_names for name in tool_names):
            return f"{agent_name}_safe_tools"

        if sensitive_names and all(name in (safe_names | sensitive_names) for name in tool_names):
            if all(name in safe_names for name in tool_names):
                return f"{agent_name}_safe_tools"
            return "pre_action_check"

        if sensitive_names:
            return "pre_action_check"
        return f"{agent_name}_safe_tools"

    return router


def _format_generic_handoff_message(current_agent: str, target: str, args: dict) -> str:
    task_summary = args.get("task_summary", "").strip()
    context = args.get("context", "").strip()
    priority = args.get("priority", "medium")

    parts = [
        f"Handoff accepted from '{current_agent}' to '{target}'.",
        f"Priority: {priority}.",
    ]
    if task_summary:
        parts.append(f"Task summary: {task_summary}")
    if context:
        parts.append(f"Context: {context}")
    return " ".join(parts)


def _format_legacy_handoff_message(current_agent: str, target: str, args: dict) -> str:
    reason = (
        args.get("request")
        or args.get("reason")
        or f"Handoff requested by {current_agent}."
    )
    return (
        f"Handoff accepted from '{current_agent}' to '{target}'. "
        f"Reason: {reason}"
    )


def handle_handoff(state: State) -> dict:
    tool_calls = _get_last_tool_calls(state)
    current_agent = state.get("active_agent", "triage")

    tool_messages = []
    handoff_history = []
    next_agent = current_agent
    pending_handoff = {}

    for tc in tool_calls:
        tool_name = tc["name"]
        args = tc.get("args", {}) or {}
        target = _parse_handoff_tool(tool_name, args)

        if target is not None:
            next_agent = target

            if tool_name == "HandoffToAgent":
                task_summary, context = augment_handoff_payload(state, target, args)
                enriched_args = dict(args)
                enriched_args["task_summary"] = task_summary
                enriched_args["context"] = context
                tool_content = _format_generic_handoff_message(current_agent, target, enriched_args)
                priority = args.get("priority", "medium")
                reason = task_summary or f"Generic handoff to {target}"
            else:
                tool_content = _format_legacy_handoff_message(current_agent, target, args)
                task_summary = args.get("request", "") or args.get("reason", "")
                context = args.get("request", "") or ""
                task_summary, context = augment_handoff_payload(
                    state,
                    target,
                    {"task_summary": task_summary, "context": context},
                )
                priority = "medium"
                reason = task_summary or f"Legacy handoff to {target}"

            tool_messages.append(
                ToolMessage(
                    content=tool_content,
                    tool_call_id=tc["id"],
                )
            )

            handoff_history.append(
                {
                    "from_agent": current_agent,
                    "to_agent": target,
                    "reason": str(reason),
                    "tool_name": tool_name,
                    "tool_call_id": tc["id"],
                }
            )

            pending_handoff = {
                "from_agent": current_agent,
                "to_agent": target,
                "task_summary": task_summary,
                "context": context,
                "priority": priority,
                "tool_name": tool_name,
                "tool_call_id": tc["id"],
                "consumed": False,
                "memory_brief": build_handoff_memory_brief(state, target),
            }
            continue

        if tool_name == CompleteOrEscalate.__name__:
            reason = args.get("reason", "Agent requested escalation / task context switch.")
            next_agent = "triage"

            tool_messages.append(
                ToolMessage(
                    content=(
                        "Escalation accepted. Return control to the triage agent "
                        f"for re-planning. Reason: {reason}"
                    ),
                    tool_call_id=tc["id"],
                )
            )

            handoff_history.append(
                {
                    "from_agent": current_agent,
                    "to_agent": "triage",
                    "reason": str(reason),
                    "tool_name": tool_name,
                    "tool_call_id": tc["id"],
                }
            )

            pending_handoff = {
                "from_agent": current_agent,
                "to_agent": "triage",
                "task_summary": reason,
                "context": build_handoff_memory_brief(state, "triage"),
                "priority": "medium",
                "tool_name": tool_name,
                "tool_call_id": tc["id"],
                "consumed": False,
                "memory_brief": build_handoff_memory_brief(state, "triage"),
            }

    return {
        "messages": tool_messages,
        "last_active_agent": current_agent,
        "active_agent": next_agent,
        "handoff_history": handoff_history,
        "pending_handoff": pending_handoff,
        "pending_tool_scope": None,
    }


def make_agent_entry(agent_name: str, assistant_callable):
    def _entry(state: State, config: RunnableConfig | None = None):
        updates = {}

        if _is_pending_handoff_for_agent(state, agent_name):
            pending = dict(state.get("pending_handoff") or {})
            pending["consumed"] = True
            updates["pending_handoff"] = pending

        result = assistant_callable(state, config=config)

        if isinstance(result, dict):
            updates.update(result)

        return updates

    return _entry


def self_reflect(state: State) -> dict:
    last_msg = _get_last_message(state)
    latest_assistant_message = getattr(last_msg, "content", "")

    result = self_reflection_chain.invoke(
        {
            "active_agent": state.get("active_agent", "triage"),
            "user_info": state.get("user_info", ""),
            "pending_handoff": state.get("pending_handoff", {}),
            "agent_brief": state.get("working_memory", {}).get("agent_brief", ""),
            "working_memory": state.get("working_memory", {}),
            "last_tool_result": state.get("last_tool_result", {}),
            "latest_assistant_message": latest_assistant_message,
        }
    )

    return {
        "reflection_trace": [
            {
                "stage": "self_reflect",
                "active_agent": state.get("active_agent", "triage"),
                "result": result.model_dump(),
            }
        ]
    }


def route_after_self_reflect(
    state: State,
) -> Literal["critic_review", "dispatch_active_agent", "write_experience_memory"]:
    trace = state.get("reflection_trace", [])
    if not trace:
        return "critic_review"

    last_result = trace[-1].get("result", {})
    status = last_result.get("status", "ok")

    if status == "ok":
        return "critic_review"
    if status == "need_user_input":
        return "write_experience_memory"

    return "dispatch_active_agent"


def critic_review(state: State) -> dict:
    last_msg = _get_last_message(state)
    latest_assistant_message = getattr(last_msg, "content", "")

    result = critic_chain.invoke(
        {
            "active_agent": state.get("active_agent", "triage"),
            "user_info": state.get("user_info", ""),
            "pending_handoff": state.get("pending_handoff", {}),
            "agent_brief": state.get("working_memory", {}).get("agent_brief", ""),
            "working_memory": state.get("working_memory", {}),
            "last_tool_result": state.get("last_tool_result", {}),
            "latest_assistant_message": latest_assistant_message,
        }
    )

    updates = {
        "reflection_trace": [
            {
                "stage": "critic_review",
                "active_agent": state.get("active_agent", "triage"),
                "result": result.model_dump(),
            }
        ]
    }

    if result.verdict == "revise" and result.revised_answer:
        updates["messages"] = [AIMessage(content=result.revised_answer)]

    return updates


def route_after_critic(
    state: State,
) -> Literal["dispatch_active_agent", "write_experience_memory"]:
    trace = state.get("reflection_trace", [])
    if not trace:
        return "write_experience_memory"

    last_result = trace[-1].get("result", {})
    verdict = last_result.get("verdict", "approve")

    return "write_experience_memory"


def pre_action_check(state: State) -> dict:
    active_agent = state.get("active_agent", "triage")
    pending_sensitive_action = _extract_sensitive_action(state, active_agent)
    latest_assistant_message = getattr(_get_last_message(state), "content", "")

    result = pre_action_chain.invoke(
        {
            "active_agent": active_agent,
            "pending_sensitive_action": pending_sensitive_action,
            "pending_handoff": state.get("pending_handoff", {}),
            "agent_brief": state.get("working_memory", {}).get("agent_brief", ""),
            "working_memory": state.get("working_memory", {}),
            "last_tool_result": state.get("last_tool_result", {}),
            "latest_assistant_message": latest_assistant_message,
        }
    )

    updates = {
        "pending_sensitive_action": pending_sensitive_action,
        "reflection_trace": [
            {
                "stage": "pre_action_check",
                "active_agent": active_agent,
                "result": result.model_dump(),
            }
        ]
    }

    if result.decision != "approve":
        missing = ", ".join(result.missing_information) if result.missing_information else "none"
        blocked_messages = []
        for tool_call in pending_sensitive_action.get("tool_calls", []):
            blocked_messages.append(
                ToolMessage(
                    content=(
                        f"Sensitive action was not executed. Decision: {result.decision}. "
                        f"Reason: {result.rationale} "
                        f"Missing information: {missing}."
                    ),
                    tool_call_id=tool_call["id"],
                )
            )
        if blocked_messages:
            updates["messages"] = blocked_messages

    return updates


def route_after_pre_action(
    state: State,
) -> Literal["update_flight_sensitive_tools", "book_car_rental_sensitive_tools", "book_hotel_sensitive_tools", "book_excursion_sensitive_tools", "dispatch_active_agent"]:
    trace = state.get("reflection_trace", [])
    if not trace:
        return "dispatch_active_agent"

    last_result = trace[-1].get("result", {})
    decision = last_result.get("decision", "revise")
    pending = state.get("pending_sensitive_action", {}) or {}
    agent = pending.get("agent")

    if decision == "approve" and agent in AGENT_REGISTRY:
        return f"{agent}_sensitive_tools"  # type: ignore[return-value]

    return "dispatch_active_agent"


def _wrap_tool_node_with_result_capture(tool_node, agent_name: str):
    def _node(state: State):
        result = tool_node.invoke(state)
        pending = state.get("pending_sensitive_action", {}) or {}
        tool_messages = []
        tool_name_by_call_id = {
            tc.get("id", ""): tc.get("name", "")
            for tc in _get_last_tool_calls(state)
        }

        updates = {}
        if isinstance(result, dict):
            updates.update(result)
            for msg in result.get("messages", []) or []:
                if getattr(msg, "type", "") == "tool":
                    tool_messages.append(
                        {
                            "tool_call_id": getattr(msg, "tool_call_id", ""),
                            "tool_name": tool_name_by_call_id.get(
                                getattr(msg, "tool_call_id", ""),
                                "",
                            ),
                            "content": getattr(msg, "content", ""),
                        }
                    )

        updates["last_tool_result"] = {
            "agent": agent_name,
            "tool_scope": "sensitive" if pending.get("agent") == agent_name else "safe",
            "pending_sensitive_action": pending,
            "tool_outputs": tool_messages,
        }

        if pending.get("agent") == agent_name:
            updates["pending_sensitive_action"] = {}

        return updates

    return _node


builder = StateGraph(State)

builder.add_node("fetch_user_info", user_info)
builder.add_node("capture_state_memory", capture_state_memory)
builder.add_node("refresh_state_memory", refresh_state_memory)
builder.add_node("bootstrap_swarm", bootstrap_swarm)
builder.add_node("retrieve_experience_memory", retrieve_experience_memory)
builder.add_node(
    "dispatch_active_agent",
    lambda state: {"active_agent": state.get("active_agent", "triage")},
)
builder.add_node("handle_handoff", handle_handoff)
builder.add_node("self_reflect", self_reflect)
builder.add_node("critic_review", critic_review)
builder.add_node("pre_action_check", pre_action_check)
builder.add_node("write_experience_memory", write_experience_memory)

builder.add_edge(START, "fetch_user_info")
builder.add_edge("fetch_user_info", "capture_state_memory")
builder.add_edge("capture_state_memory", "bootstrap_swarm")
builder.add_edge("bootstrap_swarm", "retrieve_experience_memory")
builder.add_edge("retrieve_experience_memory", "dispatch_active_agent")
builder.add_edge("refresh_state_memory", "retrieve_experience_memory")
builder.add_edge("handle_handoff", "refresh_state_memory")
builder.add_edge("write_experience_memory", END)

interrupt_nodes = []

for agent_name, spec in AGENT_REGISTRY.items():
    assistant_node = agent_name
    safe_node = f"{agent_name}_safe_tools"
    sensitive_node = f"{agent_name}_sensitive_tools"

    builder.add_node(
        assistant_node,
        make_agent_entry(agent_name, spec["assistant"]),
    )

    safe_tool_node = create_tool_node_with_fallback(spec["safe_tools"])
    builder.add_node(
        safe_node,
        _wrap_tool_node_with_result_capture(safe_tool_node, agent_name),
    )
    builder.add_edge(safe_node, "refresh_state_memory")

    if spec["sensitive_tools"]:
        sensitive_tool_node = create_tool_node_with_fallback(spec["sensitive_tools"])
        builder.add_node(
            sensitive_node,
            _wrap_tool_node_with_result_capture(sensitive_tool_node, agent_name),
        )
        builder.add_edge(sensitive_node, "refresh_state_memory")
        interrupt_nodes.append(sensitive_node)

    route_map = {
        "handle_handoff": "handle_handoff",
        "self_reflect": "self_reflect",
        "pre_action_check": "pre_action_check",
        "dispatch_active_agent": "dispatch_active_agent",
        END: END,
        safe_node: safe_node,
    }
    if spec["sensitive_tools"]:
        route_map[sensitive_node] = sensitive_node

    builder.add_conditional_edges(
        assistant_node,
        make_agent_router(agent_name),
        route_map,
    )

builder.add_conditional_edges(
    "dispatch_active_agent",
    route_dispatch_active_agent,
    {
        "triage": "triage",
        "update_flight": "update_flight",
        "book_car_rental": "book_car_rental",
        "book_hotel": "book_hotel",
        "book_excursion": "book_excursion",
    },
)

builder.add_conditional_edges(
    "self_reflect",
    route_after_self_reflect,
    {
        "critic_review": "critic_review",
        "dispatch_active_agent": "dispatch_active_agent",
        "write_experience_memory": "write_experience_memory",
    },
)

builder.add_conditional_edges(
    "critic_review",
    route_after_critic,
    {
        "dispatch_active_agent": "dispatch_active_agent",
        "write_experience_memory": "write_experience_memory",
    },
)

builder.add_conditional_edges(
    "pre_action_check",
    route_after_pre_action,
    {
        "update_flight_sensitive_tools": "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools": "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools": "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools": "book_excursion_sensitive_tools",
        "dispatch_active_agent": "dispatch_active_agent",
    },
)

memory = build_checkpointer()

multi_agentic_graph = builder.compile(
    checkpointer=memory,
    interrupt_before=interrupt_nodes,
)
