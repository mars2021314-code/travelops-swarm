from __future__ import annotations

import asyncio
import threading
from typing import Any

from customer_support_chat.app.services.mcp.registry import AGENT_MCP_ACCESS
from customer_support_chat.app.services.mcp.tool_loader import (
    load_all_mcp_tools,
    load_mcp_tools_by_server,
)


_MCP_TOOLS_CACHE_ALL: list[Any] | None = None
_MCP_TOOLS_CACHE_BY_AGENT: dict[str, list[Any]] = {}

_MCP_CACHE_LOCK = threading.Lock()


def _run_async(coro):
    """
    Run async code safely from sync code.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container: dict[str, Any] = {}
    error_container: dict[str, BaseException] = {}

    def runner():
        try:
            result_container["result"] = asyncio.run(coro)
        except BaseException as e:
            error_container["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()

    if "error" in error_container:
        raise error_container["error"]

    return result_container.get("result")


def get_cached_mcp_tools(strict: bool = False) -> list[Any]:
    """
    Load and cache all MCP tools.
    """
    global _MCP_TOOLS_CACHE_ALL

    if _MCP_TOOLS_CACHE_ALL is not None:
        return _MCP_TOOLS_CACHE_ALL

    with _MCP_CACHE_LOCK:
        if _MCP_TOOLS_CACHE_ALL is not None:
            return _MCP_TOOLS_CACHE_ALL

        try:
            tools = _run_async(load_all_mcp_tools())
            _MCP_TOOLS_CACHE_ALL = tools or []
            return _MCP_TOOLS_CACHE_ALL
        except Exception:
            if strict:
                raise
            _MCP_TOOLS_CACHE_ALL = []
            return _MCP_TOOLS_CACHE_ALL


def get_cached_mcp_tools_for_agent(agent_name: str, strict: bool = False) -> list[Any]:
    """
    Load and cache only the MCP tools the given agent is allowed to use.
    """
    if agent_name in _MCP_TOOLS_CACHE_BY_AGENT:
        return _MCP_TOOLS_CACHE_BY_AGENT[agent_name]

    allowed_servers = AGENT_MCP_ACCESS.get(agent_name, [])

    with _MCP_CACHE_LOCK:
        if agent_name in _MCP_TOOLS_CACHE_BY_AGENT:
            return _MCP_TOOLS_CACHE_BY_AGENT[agent_name]

        try:
            if not allowed_servers:
                _MCP_TOOLS_CACHE_BY_AGENT[agent_name] = []
                return []

            tools = _run_async(load_mcp_tools_by_server(allowed_servers))
            _MCP_TOOLS_CACHE_BY_AGENT[agent_name] = tools or []
            return _MCP_TOOLS_CACHE_BY_AGENT[agent_name]
        except Exception:
            if strict:
                raise
            _MCP_TOOLS_CACHE_BY_AGENT[agent_name] = []
            return []


def refresh_cached_mcp_tools(strict: bool = False) -> dict[str, list[Any]]:
    """
    Force refresh both global and per-agent MCP caches.
    """
    global _MCP_TOOLS_CACHE_ALL, _MCP_TOOLS_CACHE_BY_AGENT

    with _MCP_CACHE_LOCK:
        _MCP_TOOLS_CACHE_ALL = None
        _MCP_TOOLS_CACHE_BY_AGENT = {}

    refreshed = {
        "__all__": get_cached_mcp_tools(strict=strict),
    }

    for agent_name in AGENT_MCP_ACCESS:
        refreshed[agent_name] = get_cached_mcp_tools_for_agent(agent_name, strict=strict)

    return refreshed


def clear_cached_mcp_tools() -> None:
    """
    Clear MCP caches without reloading.
    """
    global _MCP_TOOLS_CACHE_ALL, _MCP_TOOLS_CACHE_BY_AGENT

    with _MCP_CACHE_LOCK:
        _MCP_TOOLS_CACHE_ALL = None
        _MCP_TOOLS_CACHE_BY_AGENT = {}