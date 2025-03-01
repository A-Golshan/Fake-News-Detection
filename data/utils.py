import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import HeteroData

def step(data, t):

    new_data = HeteroData(
        y=data.y
    )

    for node_type in data.node_types:
        new_data[node_type].x = data[node_type].x

    for edge_type in data.edge_types:
        mask = data[edge_type].time < t
        new_data[edge_type].edge_index = data[edge_type].edge_index[:, mask]
        new_data[edge_type].time = data[edge_type].time[mask]

    return new_data

def plot(
    G: nx.classes.digraph.DiGraph, 
    color_mapping: dict, 
    layout: dict, 
    label: str, 
    node_size: int=50
):

    node_colors = [color_mapping[G.nodes[node]['type']] for node in G.nodes]
    node_borders = 'black'
    node_sizes = [node_size + 20 * G.degree(node) for node in G.nodes]

    nx.draw_networkx_nodes(
        G, pos=layout, node_color=node_colors, edgecolors=node_borders,
        node_size=node_sizes, linewidths=0.5
    )

    nx.draw_networkx_edges(G, pos=layout, alpha=0.6, width=1)

    plt.scatter([], [], c=color_mapping['user'], label='User')
    plt.scatter([], [], c=color_mapping['tweet'], label='Tweet')
    plt.plot([], [], c='black', label='Edge')
    plt.plot([], [], c='white', label=f'Label: {label}')
    plt.legend()