from __future__ import annotations

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate

from customer_support_chat.app.services.assistants.assistant_base import llm
from customer_support_chat.app.services.reflection.schemas import CriticDecision


critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the final critic for a swarm-based travel support assistant.\n\n"
            "Your job is to review the latest assistant reply before it is shown to the user.\n\n"
            "Critic rules:\n"
            "- Reject answers that claim a booking/update/cancellation succeeded without tool evidence.\n"
            "- Reject answers that contradict the latest tool result, working memory, or handoff context.\n"
            "- Treat structured current-state facts in working_memory as stronger evidence than candidate options or historical experience.\n"
            "- Reject answers that present candidate options as if they were already executed choices.\n"
            "- Reject answers that present policy evidence as if it were proof of a completed transactional action.\n"
            "- Reject answers that hide important uncertainty.\n"
            "- Reject answers that ask the user to repeat information already present in handoff context or working memory.\n"
            "- Approve answers that are grounded, coherent, and honest about what was or was not completed.\n\n"

            "If revision is needed, provide a corrected user-facing answer.\n\n"
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
            "Latest assistant reply:\n{latest_assistant_message}\n\n"
            "Return a structured CriticDecision."
        ),
    ]
).partial(time=datetime.now())


critic_chain = critic_prompt | llm.with_structured_output(
    CriticDecision,
    method="function_calling",
)
