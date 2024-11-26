import os
import random
import networkx as nx
import numpy as np
import torch

from scipy import sparse as sp
from torch_geometric.data import Data

def load_data(dataset_name):
    communties = open(f"./Datasets/{dataset_name}/{dataset_name}.cmty.txt")
    edges = open(f"./Datasets/{dataset_name}/{dataset_name}.ungraph.txt")

    communties = [[int(i) for i in x.split()] for x in communties]
    edges = [[int(i) for i in e.split()] for e in edges]
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]

    nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}

    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communties = [[mapping[node] for node in com] for com in communties]

    num_node, num_edges, num_comm = len(nodes), len(edges), len(communties)

    if os.path.exists(f"./Datasets/{dataset_name}/{dataset_name}.nodefeat.txt"):
        with open(f"./Datasets/{dataset_name}/{dataset_name}.nodefeat.txt") as fh:
            nodefeats = [x.split() for x in fh.read().strip().split('\n')]
            nodefeats = {int(k): [int(i) for i in v] for k, *v in nodefeats}
            ind = np.array([[i, j] for i, js in nodefeats.items() for j in js])
            nodefeats = sp.csr_matrix((np.ones(len(ind)), (ind[:, 0], ind[:, 1]))).todense().A
    else:
        nodefeats = None

    nx_graph = nx.Graph(edges)
    nx_graph.add_nodes_from(nodes)

    lengths = [len(arr) for arr in communties]
    average_length = round(np.mean(lengths),2)

    degree = [degree for node, degree in nx_graph.degree()]
    average_degree = sum(degree) / len(degree)

    print(f"#Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}, #Avg_comms {average_length}, #Avg_degree {average_degree}")

    return num_node, num_edges, num_comm, nodes, edges, communties, nodefeats, nx_graph

def feature_generation(num_node, communities):
    num_attribute = int(0.01 * num_node)
    nodefeats = np.zeros([num_node, num_attribute])

    nodes = [set() for i in range(num_node)]

    all_attributes = list(range(num_attribute))

    for community in communities:
        assign_attributes_to_community(nodes, community, all_attributes)

    add_random_noise(nodes, num_node)

    for i in range(num_node):
        for attr in nodes[i]:
            nodefeats[i, attr] = 1

    return nodefeats

def assign_attributes_to_community(nodes, community, attributes, p=0.8):
    num_vertices = len(community)
    selected_vertices = random.sample(community, int(p * num_vertices))
    selected_attributes = random.sample(attributes, 3)

    for v in selected_vertices:
        for attr in selected_attributes:
            nodes[v].add(attr)

def add_random_noise(nodes, num_node):
    for i in range(num_node):
        noise_attr = random.choice([1,2,3,4,5])
        nodes[i].add(noise_attr)

def prepare_data(dataset):
    print(f"Start loading {dataset.upper()} dataset...")

    num_node, num_edge, num_comm, nodes, edges, communities, nodefeats, nx_graph = load_data(dataset)

    if nodefeats is None:
        nodefeats = feature_generation(num_node, communities)

    feat = (nodefeats - nodefeats.mean(0, keepdims=True)) / (nodefeats.std(0, keepdims=True) + 1e-9)

    print(f"#Attributes {nodefeats.shape[1]}")

    converted_edges = [[v, u] for u, v in edges]
    graph_data = Data(x=torch.FloatTensor(feat), edge_index=torch.LongTensor(np.array(edges + converted_edges)).transpose(0, 1))

    print("")

    return num_node, num_edge, num_comm, nx_graph, nodefeats, graph_data, communities