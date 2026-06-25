from __future__ import annotations

from typing import Any

from customer_support_chat.app.services.mcp.client import get_mcp_client


async def load_all_mcp_tools() -> list[Any]:
    """
    Load all tools exposed by configured MCP servers.
    """
    client = get_mcp_client()
    tools = await client.get_tools()
    return list(tools or [])


def _tool_matches_server(tool: Any, server_name: str) -> bool:
    """
    Conservative matching for MCP tool names.

    Depending on adapter/server setup, tool names may look like:
    - policy_server.lookup_policy_topic
    - policy_server_lookup_policy_topic
    - policy_server:lookup_policy_topic
    - lookup_policy_topic   (least desirable / ambiguous case)

    We primarily rely on prefixed names.
    """
    tool_name = getattr(tool, "name", "") or ""
    return (
        tool_name.startswith(f"{server_name}.")
        or tool_name.startswith(f"{server_name}_")
        or tool_name.startswith(f"{server_name}:")
    )


async def load_mcp_tools_by_server(server_names: list[str]) -> list[Any]:
    """
    Load all MCP tools, then filter by allowed server names.
    """
    if not server_names:
        return []

    all_tools = await load_all_mcp_tools()
    filtered: list[Any] = []

    for tool in all_tools:
        if any(_tool_matches_server(tool, server_name) for server_name in server_names):
            filtered.append(tool)

    return filtered