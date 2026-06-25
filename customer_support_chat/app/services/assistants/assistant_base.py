from __future__ import annotations

from typing import Literal, Optional

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from customer_support_chat.app.core.settings import get_settings

# Get settings
settings = get_settings()

# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------

llm = ChatOpenAI(
    model=settings.DEEPSEEK_MODEL,
    temperature=0,
    openai_api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    timeout=settings.LLM_TIMEOUT_SECONDS,
    max_retries=settings.LLM_MAX_RETRIES,
)


# ---------------------------------------------------------------------
# Agent names / registry-level constants
# ---------------------------------------------------------------------

AgentName = Literal[
    "triage",
    "update_flight",
    "book_car_rental",
    "book_hotel",
    "book_excursion",
    "critic",
    "memory",
]

AGENT_DESCRIPTIONS = {
    "triage": "General support, policy lookup, lightweight search, and routing to the best peer agent.",
    "update_flight": "Flight change / cancellation workflows and flight-related transactional operations.",
    "book_car_rental": "Car-rental search, booking, update, and cancellation workflows.",
    "book_hotel": "Hotel search, booking, update, and cancellation workflows.",
    "book_excursion": "Excursion / activity recommendation, booking, update, and cancellation workflows.",
    "critic": "Verification, consistency checking, and self-correction support.",
    "memory": "Long-term memory / experience retrieval and write-back support.",
}


def make_handoff_payload_hint() -> str:
    return (
        "When handing off, include a concise summary of user intent, "
        "what has already been done, what remains to do, and any key constraints. "
        "If available, mention current confirmed facts, relevant candidate options, "
        "recent policy findings, and unresolved open loops."
    )


# ---------------------------------------------------------------------
# Generic handoff schema
# ---------------------------------------------------------------------

class HandoffToAgent(BaseModel):
    """
    Generic peer-to-peer handoff tool for swarm routing.

    Use this whenever another peer agent is the better owner of the next step.
    """

    target_agent: AgentName = Field(
        description=(
            "The peer agent that should take over. "
            "Typical values: update_flight, book_car_rental, book_hotel, book_excursion."
        )
    )
    task_summary: str = Field(
        description=(
            "A concise summary of what the next agent should do, including the user's goal."
        )
    )
    context: str = Field(
        description=(
            "Relevant context for the next agent: what has already been checked, "
            "important entities, dates, constraints, preferences, and unresolved questions."
        )
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Operational priority for the handoff."
    )


# ---------------------------------------------------------------------
# Compatibility fallback
# ---------------------------------------------------------------------

class CompleteOrEscalate(BaseModel):
    """Compatibility fallback when the current agent should stop and let triage re-plan."""

    cancel: bool = True
    reason: str = Field(
        description=(
            "Why the current agent cannot or should not continue. "
            "Use this only as a fallback if direct peer handoff is not appropriate."
        )
    )


# ---------------------------------------------------------------------
# Assistant wrapper
# ---------------------------------------------------------------------

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state, config: Optional[RunnableConfig] = None):
        while True:
            result = self.runnable.invoke(state, config=config)

            # If the model returned an empty response, nudge it once more.
            if not result.tool_calls and (
                not result.content
                or (isinstance(result.content, list) and not result.content[0].get("text"))
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break

        return {"messages": result}
