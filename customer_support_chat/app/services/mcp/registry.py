from __future__ import annotations

from typing import Dict, Any
import threading

MCP_SERVER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "policy_server": {
        "transport": "stdio",
        "command": "python",
        "args": [
            "-m",
            "customer_support_chat.app.services.mcp.servers.policy_server",
        ],
        "description": "Policy lookup and refund-rule related MCP server.",
    },
    # 以后可继续扩展
    # "hotel_ops_server": {...}
    # "rental_ops_server": {...}
}

AGENT_MCP_ACCESS: Dict[str, list[str]] = {
    "triage": ["policy_server"],
    "update_flight": ["policy_server"],
    "book_hotel": [],
    "book_car_rental": [],
    "book_excursion": [],
}

_registry_lock = threading.Lock()

def register_mcp_server(server_name: str, config: Dict[str, Any]) -> None:
    """
    Dynamically register a new MCP server at runtime.
    
    Args:
        server_name: Unique name for the server
        config: Server configuration dict with transport, command, args, etc.
    """
    with _registry_lock:
        if server_name in MCP_SERVER_REGISTRY:
            raise ValueError(f"MCP server '{server_name}' already registered")
        MCP_SERVER_REGISTRY[server_name] = config

def unregister_mcp_server(server_name: str) -> None:
    """
    Remove a dynamically registered MCP server.
    """
    with _registry_lock:
        if server_name not in MCP_SERVER_REGISTRY:
            raise ValueError(f"MCP server '{server_name}' not found")
        del MCP_SERVER_REGISTRY[server_name]
        
        # Also remove from access control
        for agent, servers in AGENT_MCP_ACCESS.items():
            if server_name in servers:
                servers.remove(server_name)

def grant_agent_mcp_access(agent_name: str, server_names: list[str]) -> None:
    """
    Grant an agent access to specific MCP servers.
    
    Args:
        agent_name: Name of the agent
        server_names: List of server names to grant access to
    """
    with _registry_lock:
        if agent_name not in AGENT_MCP_ACCESS:
            AGENT_MCP_ACCESS[agent_name] = []
        for server in server_names:
            if server not in AGENT_MCP_ACCESS[agent_name]:
                AGENT_MCP_ACCESS[agent_name].append(server)

def revoke_agent_mcp_access(agent_name: str, server_names: list[str]) -> None:
    """
    Revoke an agent's access to specific MCP servers.
    """
    with _registry_lock:
        if agent_name in AGENT_MCP_ACCESS:
            for server in server_names:
                if server in AGENT_MCP_ACCESS[agent_name]:
                    AGENT_MCP_ACCESS[agent_name].remove(server)