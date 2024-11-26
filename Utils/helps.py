import copy

import networkx as nx
import numpy as np
import random

def choose_seed(com, nx_graph):
    subgraph = nx_graph.subgraph(com)
    # degree = nx.degree(subgraph)
    # nodes = sorted(list(subgraph.nodes()))
    # node2degree = [degree[node] for node in nodes]
    # degree = np.array(node2degree)
    # if not all(map(lambda x: x == 0, degree)):
    #     degree = degree / np.sum(degree, axis=0)
    #     seed = np.random.choice(nodes, p=degree.ravel())
    # else:
    #     seed = np.random.choice(nodes)
    seed = max(subgraph.nodes(), key=subgraph.degree)

    return seed

def split_communities(num_community, train_size, valid_size):
    print("Start spliting communities...")

    idxes = list(range(num_community))

    train_idx = random.sample(idxes, k=train_size)
    remains = list(set(idxes) - set(train_idx))
    valid_idx = random.sample(remains, k=valid_size)
    test_idx = list(set(remains) - set(valid_idx))

    print(f"#Train {len(train_idx)}, #Validation {len(valid_idx)}, #Test {len(test_idx)}")

    print("")

    return train_idx, train_idx, train_idx

def generate_outer_boundary(graph, com_nodes, com, max_size=20):
    outer_nodes = []
    for node in com_nodes:
        outer_nodes += list(graph.neighbors(node))
    outer_nodes = list(set(outer_nodes) - set(com_nodes))
    if len(outer_nodes) > max_size:
        true = sorted(list(set(outer_nodes) & set(com)))
        false = sorted(list(set(outer_nodes) - set(true)))
        true = true if len(true) <= 10 else true[:10]
        false = false[:max_size - len(true)]
        outer_nodes = true + false

    return outer_nodes

def generate_inner_boundary(graph, com_nodes, com, max_size=20):
    inner_nodes = copy.deepcopy(com_nodes)
    if len(inner_nodes) > max_size:
        false = sorted(list(set(inner_nodes) & set(com)))
        true = sorted(list(set(inner_nodes) - set(false)))
        true = true if len(true) <= 10 else true[:10]
        false = false[:max_size - len(true)]
        inner_nodes = true + false

    return inner_nodes