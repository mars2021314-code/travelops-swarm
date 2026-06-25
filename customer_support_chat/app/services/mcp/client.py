from __future__ import annotations

from functools import lru_cache

from langchain_mcp_adapters.client import MultiServerMCPClient

from customer_support_chat.app.services.mcp.registry import MCP_SERVER_REGISTRY


def _build_client_config() -> dict:
    config: dict = {}

    for server_name, spec in MCP_SERVER_REGISTRY.items():
        transport = spec["transport"]

        if transport == "stdio":
            config[server_name] = {
                "transport": "stdio",
                "command": spec["command"],
                "args": spec.get("args", []),
            }
        elif transport in {"sse", "streamable_http"}:
            config[server_name] = {
                "transport": transport,
                "url": spec["url"],
            }
        else:
            raise ValueError(f"Unsupported MCP transport for {server_name}: {transport}")

    return config


@lru_cache(maxsize=1)
def get_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(_build_client_config())