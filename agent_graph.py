import matplotlib.pyplot as plt
import networkx as nx


def build_graph():
    G = nx.DiGraph()

    # =========================
    # 1) Nodes
    # =========================
    nodes = {
        "START": "start_end",
        "END": "start_end",
        "fetch_user_info": "core",
        "bootstrap_swarm": "core",
        "retrieve_experience_memory": "core",
        "dispatch_active_agent": "decision",
        "handle_handoff": "core",
        "self_reflect": "core",
        "critic_review": "core",
        "pre_action_check": "core",
        "write_experience_memory": "core",
        "route_after_self_reflect": "decision",
        "route_after_critic": "decision",
        "route_after_pre_action": "decision",
        "triage": "agent",
        "triage_safe_tools": "safe_tool",
        "update_flight": "agent",
        "update_flight_safe_tools": "safe_tool",
        "update_flight_sensitive_tools": "sensitive_tool",
        "book_car_rental": "agent",
        "book_car_rental_safe_tools": "safe_tool",
        "book_car_rental_sensitive_tools": "sensitive_tool",
        "book_hotel": "agent",
        "book_hotel_safe_tools": "safe_tool",
        "book_hotel_sensitive_tools": "sensitive_tool",
        "book_excursion": "agent",
        "book_excursion_safe_tools": "safe_tool",
        "book_excursion_sensitive_tools": "sensitive_tool",
    }

    for node, node_type in nodes.items():
        G.add_node(node, node_type=node_type)

    # =========================
    # 2) Edges
    # =========================
    edges = [
        ("START", "fetch_user_info", ""),
        ("fetch_user_info", "bootstrap_swarm", ""),
        ("bootstrap_swarm", "retrieve_experience_memory", ""),
        ("retrieve_experience_memory", "dispatch_active_agent", ""),
        ("handle_handoff", "dispatch_active_agent", ""),
        ("write_experience_memory", "END", ""),

        ("dispatch_active_agent", "triage", "triage"),
        ("dispatch_active_agent", "update_flight", "update_flight"),
        ("dispatch_active_agent", "book_car_rental", "book_car_rental"),
        ("dispatch_active_agent", "book_hotel", "book_hotel"),
        ("dispatch_active_agent", "book_excursion", "book_excursion"),

        ("triage", "handle_handoff", "handoff / escalate"),
        ("triage", "self_reflect", "no tools / END"),
        ("triage", "triage_safe_tools", "all safe tools"),
        ("triage_safe_tools", "dispatch_active_agent", "tool result"),

        ("update_flight", "handle_handoff", "handoff / escalate"),
        ("update_flight", "self_reflect", "no tools / END"),
        ("update_flight", "update_flight_safe_tools", "all safe tools"),
        ("update_flight", "pre_action_check", "contains sensitive tools"),
        ("update_flight_safe_tools", "dispatch_active_agent", "tool result"),
        ("update_flight_sensitive_tools", "dispatch_active_agent", "tool result"),

        ("book_car_rental", "handle_handoff", "handoff / escalate"),
        ("book_car_rental", "self_reflect", "no tools / END"),
        ("book_car_rental", "book_car_rental_safe_tools", "all safe tools"),
        ("book_car_rental", "pre_action_check", "contains sensitive tools"),
        ("book_car_rental_safe_tools", "dispatch_active_agent", "tool result"),
        ("book_car_rental_sensitive_tools", "dispatch_active_agent", "tool result"),

        ("book_hotel", "handle_handoff", "handoff / escalate"),
        ("book_hotel", "self_reflect", "no tools / END"),
        ("book_hotel", "book_hotel_safe_tools", "all safe tools"),
        ("book_hotel", "pre_action_check", "contains sensitive tools"),
        ("book_hotel_safe_tools", "dispatch_active_agent", "tool result"),
        ("book_hotel_sensitive_tools", "dispatch_active_agent", "tool result"),

        ("book_excursion", "handle_handoff", "handoff / escalate"),
        ("book_excursion", "self_reflect", "no tools / END"),
        ("book_excursion", "book_excursion_safe_tools", "all safe tools"),
        ("book_excursion", "pre_action_check", "contains sensitive tools"),
        ("book_excursion_safe_tools", "dispatch_active_agent", "tool result"),
        ("book_excursion_sensitive_tools", "dispatch_active_agent", "tool result"),

        ("self_reflect", "route_after_self_reflect", ""),
        ("route_after_self_reflect", "critic_review", "status = ok"),
        ("route_after_self_reflect", "dispatch_active_agent", "status != ok"),

        ("critic_review", "route_after_critic", ""),
        ("route_after_critic", "write_experience_memory", "verdict = approve"),
        ("route_after_critic", "dispatch_active_agent", "verdict != approve"),

        ("pre_action_check", "route_after_pre_action", ""),
        ("route_after_pre_action", "update_flight_sensitive_tools", "approve + update_flight"),
        ("route_after_pre_action", "book_car_rental_sensitive_tools", "approve + book_car_rental"),
        ("route_after_pre_action", "book_hotel_sensitive_tools", "approve + book_hotel"),
        ("route_after_pre_action", "book_excursion_sensitive_tools", "approve + book_excursion"),
        ("route_after_pre_action", "dispatch_active_agent", "reject / revise"),
    ]

    for u, v, label in edges:
        G.add_edge(u, v, label=label)

    return G


def get_positions():
    # 手工分层布局，便于阅读
    pos = {
        "START": (0, 6),
        "fetch_user_info": (2, 6),
        "bootstrap_swarm": (4, 6),
        "retrieve_experience_memory": (6, 6),
        "dispatch_active_agent": (8, 6),

        "triage": (11, 9),
        "triage_safe_tools": (14, 9),

        "update_flight": (11, 7),
        "update_flight_safe_tools": (14, 7.6),
        "update_flight_sensitive_tools": (14, 6.4),

        "book_car_rental": (11, 5),
        "book_car_rental_safe_tools": (14, 5.6),
        "book_car_rental_sensitive_tools": (14, 4.4),

        "book_hotel": (11, 3),
        "book_hotel_safe_tools": (14, 3.6),
        "book_hotel_sensitive_tools": (14, 2.4),

        "book_excursion": (11, 1),
        "book_excursion_safe_tools": (14, 1.6),
        "book_excursion_sensitive_tools": (14, 0.4),

        "handle_handoff": (17, 9),
        "pre_action_check": (17, 5),
        "route_after_pre_action": (20, 5),

        "self_reflect": (17, 2),
        "route_after_self_reflect": (20, 2),
        "critic_review": (23, 2),
        "route_after_critic": (26, 2),
        "write_experience_memory": (29, 2),
        "END": (32, 2),
    }
    return pos


def draw_graph(G):
    pos = get_positions()

    color_map = {
        "start_end": "#d9eaf7",
        "core": "#e8f5e9",
        "agent": "#fff3cd",
        "safe_tool": "#e3f2fd",
        "sensitive_tool": "#fdecea",
        "decision": "#f3e5f5",
    }

    node_colors = [color_map[G.nodes[n]["node_type"]] for n in G.nodes()]

    plt.figure(figsize=(24, 10))
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=200,
        edgecolors="black",
        linewidths=1.0
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_family="sans-serif"
    )
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=18,
        width=1.2,
        connectionstyle="arc3,rad=0.05"
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    edge_labels = {k: v for k, v in edge_labels.items() if v}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        rotate=False,
        bbox=dict(alpha=0.7, color="white", edgecolor="none")
    )

    plt.title("Multi-Agent RAG Customer Support Graph", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    G = build_graph()
    draw_graph(G)