from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("policy_server")


POLICY_DB = {
    "refund": {
        "summary": "Refund eligibility depends on fare conditions and timing before departure.",
        "rules": [
            "Non-refundable fares may still allow taxes/fees recovery depending on policy.",
            "Refundability should be checked before cancellation is executed.",
            "Cancellation eligibility does not guarantee full refund eligibility.",
        ],
    },
    "cancellation": {
        "summary": "Cancellation may incur penalties depending on fare rules and timing.",
        "rules": [
            "Check fare conditions before promising a refund.",
            "User confirmation is recommended before irreversible cancellation actions.",
        ],
    },
    "delay": {
        "summary": "Delays may require itinerary-wide coordination across flight, hotel, rental car, and excursions.",
        "rules": [
            "Do not assume downstream bookings remain valid after a major delay.",
            "Confirm whether hotel, rental, and excursion timings need adjustment.",
        ],
    },
}


@mcp.tool()
def lookup_policy_topic(
    topic: Literal["refund", "cancellation", "delay"],
) -> str:
    """
    Look up a structured travel-support policy topic.
    """
    item = POLICY_DB.get(topic)
    if not item:
        return f"No policy found for topic={topic}"

    lines = [f"Topic: {topic}", f"Summary: {item['summary']}", "Rules:"]
    lines.extend([f"- {r}" for r in item["rules"]])
    return "\n".join(lines)


@mcp.tool()
def assess_refund_risk(
    fare_refundable: bool,
    hours_before_departure: int,
) -> str:
    """
    Return a lightweight refund-risk assessment.
    """
    if fare_refundable:
        return (
            "Refund-risk assessment: likely refundable.\n"
            "Still verify detailed fare conditions and any timing-specific penalties."
        )

    if hours_before_departure < 24:
        return (
            "Refund-risk assessment: high risk of penalty or no refund.\n"
            "Do not promise refundability before checking detailed fare rules."
        )

    return (
        "Refund-risk assessment: uncertain / likely restricted.\n"
        "Check detailed policy before cancellation."
    )


if __name__ == "__main__":
    mcp.run()