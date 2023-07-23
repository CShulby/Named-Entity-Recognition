import networkx as nx
import random
import matplotlib.pyplot as plt


def generate_knowledge_graph(entities, relations):
    """
    Generate a knowledge graph based on the extracted entities and relations.

    Args:
        entities (list): List of tuples containing extracted entities and their labels.
        relations (list): List of tuples containing extracted relations between entities.

    Returns:
        nx.DiGraph: A directed graph representing the knowledge graph.
    """
    G = nx.DiGraph()

    # Adding entities as nodes with random colors based on entity types
    entity_types = set([label for _, label in entities])
    entity_colors = {entity_type: random_color() for entity_type in entity_types}

    for entity, label in entities:
        entity_type = label.split("-")[1] if "-" in label else label
        color = entity_colors.get(entity_type, random_color())
        G.add_node(entity, label=label, entity_type=entity_type, color=color)

    # Adding relations as edges with labels and connections
    for relation in relations:
        if len(relation) == 3:
            entity1, rel, entity2 = relation
            connection = (entity1, rel, entity2)
            G.add_edge(entity1, entity2, label=rel, connection=connection)

    # Assign random colors to nodes without a color attribute
    node_colors = [G.nodes[node].get("color", random_color()) for node in G.nodes]
    nx.set_node_attributes(G, dict(zip(G.nodes, node_colors)), "color")

    return G


def random_color():
    """
    Generate a random hexadecimal color code.

    Returns:
        str: Hexadecimal color code (e.g., "#RRGGBB").
    """
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def plot_knowledge_graph(G):
    """
    Plot the knowledge graph using NetworkX and Matplotlib.

    Args:
        G (nx.DiGraph): The knowledge graph as a directed graph.

    Returns:
        None (Displays the plot).
    """
    pos = nx.shell_layout(G)  # Use shell_layout or spectral_layout
    plt.figure(figsize=(10, 8))

    # Draw nodes with colors based on entity types
    node_colors = [G.nodes[node]["color"] for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors)

    # Draw edges with labels and arrows
    edge_labels = {(u, v): G.edges[u, v]["label"] for (u, v) in G.edges}
    nx.draw_networkx_edges(
        G, pos, arrowstyle="->", arrowsize=50, width=1.0
    )  # Updated parameters here

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Draw node labels
    node_labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

    plt.axis("off")
    plt.show()
