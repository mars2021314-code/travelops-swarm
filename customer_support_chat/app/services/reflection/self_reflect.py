from __future__ import annotations

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate

from customer_support_chat.app.services.assistants.assistant_base import llm
from customer_support_chat.app.services.reflection.schemas import ReflectionDecision


self_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a lightweight self-reflection module for a swarm-based travel support system.\n\n"
            "Your task is to inspect the current conversation state and the latest assistant output, "
            "then decide whether the current response/action is ready, needs revision, "
            "needs another tool call, needs a handoff, or needs more user input.\n\n"

            "Reflection criteria:\n"
            "- Do not approve claims that an action is completed unless there is tool evidence.\n"
            "- Prefer using existing handoff context and working memory before asking the user to repeat information.\n"
            "- Treat current-state facts in working_memory as stronger evidence than candidate options or generic historical experiences.\n"
            "- Treat candidate options as possibilities only; they do not prove an action was executed.\n"
            "- Treat policy snapshots as rule evidence only; they do not prove a booking/update/cancellation occurred.\n"
            "- If the current agent is the wrong owner for the next step, recommend need_handoff.\n"
            "- If key structured facts are missing and cannot be recovered from state, recommend need_user_input.\n"
            "- If more evidence is needed from tools, recommend need_tool.\n"
            "- If the answer is mostly correct but wording or claims need adjustment, recommend revise.\n"
            "- If the answer/action is acceptable, return ok.\n\n"

            "Current time: {time}."
        ),
        (
            "human",
            "Active agent: {active_agent}\n\n"
            "Current user flight information:\n{user_info}\n\n"
            "Pending handoff: {pending_handoff}\n\n"
            "Agent brief: {agent_brief}\n\n"
            "Working memory: {working_memory}\n\n"
            "Last tool result: {last_tool_result}\n\n"
            "Latest assistant message:\n{latest_assistant_message}\n\n"
            "Return a structured ReflectionDecision."
        ),
    ]
).partial(time=datetime.now())


self_reflection_chain = self_reflection_prompt | llm.with_structured_output(
    ReflectionDecision,
    method="function_calling",
)
