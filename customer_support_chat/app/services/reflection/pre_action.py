from __future__ import annotations

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate

from customer_support_chat.app.services.assistants.assistant_base import llm
from customer_support_chat.app.services.reflection.schemas import PreActionDecision


pre_action_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a pre-action execution gate for a swarm-based travel support assistant.\n\n"
            "Your task is to evaluate whether a pending sensitive tool call should proceed.\n\n"

            "Decision policy:\n"
            "- approve: execute the sensitive action now.\n"
            "- ask_user: more explicit user confirmation or missing user choice is needed.\n"
            "- need_tool: more evidence should be collected via non-sensitive tools first.\n"
            "- revise: the current planned action is poorly formed or makes unsupported assumptions.\n"
            "- handoff: another peer agent should own the next step instead.\n\n"

            "Approval rules:\n"
            "- Do not approve if required parameters are clearly missing.\n"
            "- Do not approve if the action claims certainty without tool evidence.\n"
            "- Treat candidate options as possible inputs for the next action, not as proof that the user already selected one.\n"
            "- Treat policy snapshots as decision support, not as proof that an action has already been executed or approved by the user.\n"
            "- Do not approve if the user has not made the necessary choice among options.\n"
            "- Prefer ask_user when there is an unresolved user preference or confirmation requirement.\n"
            "- Prefer need_tool when additional search or policy lookup should happen before executing.\n"
            "- Prefer handoff when another domain agent is clearly the better owner.\n\n"

            "Current time: {time}."
        ),
        (
            "human",
            "Active agent: {active_agent}\n\n"
            "Pending sensitive action: {pending_sensitive_action}\n\n"
            "Pending handoff: {pending_handoff}\n\n"
            "Agent brief: {agent_brief}\n\n"
            "Working memory: {working_memory}\n\n"
            "Last tool result: {last_tool_result}\n\n"
            "Latest assistant message:\n{latest_assistant_message}\n\n"
            "Return a structured PreActionDecision."
        ),
    ]
).partial(time=datetime.now())


pre_action_chain = pre_action_prompt | llm.with_structured_output(
    PreActionDecision,
    method="function_calling",
)
