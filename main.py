import argparse
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import time

from Extraction.extraction import Extraction
from Refinement.Encoder.encoder import Encoder
from Refinement.Refiner.refiner import Refiner
from Utils.helps import split_communities
from Utils.load_dataset import prepare_data
from Utils.metrics import matrics, boxline


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    seed_all(args.seed)

    num_node, num_edge, num_community, nx_graph, nodefeats, graph_data, communities = prepare_data(args.dataset)
    train_idx, valid_idx, test_idx = split_communities(num_community, args.train_size, args.valid_size)

    train_comms = [communities[i] for i in train_idx]
    valid_comms = [communities[i] for i in valid_idx]
    test_comms = [communities[i] for i in test_idx]

    # Adaptive Community Extraction
    extraction = Extraction(args, nx_graph, nodefeats, communities)
    pre_coms = extraction.extract()

    train_pre_coms = [pre_coms[i] for i in train_idx]
    valid_pre_coms = [pre_coms[i] for i in valid_idx]
    test_pre_coms = [pre_coms[i] for i in test_idx]

    # Flexible Community Refinement
    # -------------------------------------------State encoder------------------------------------------------
    encoder = Encoder(args, nx_graph, graph_data, train_comms)
    encoder.train()
    embedding = encoder.get_embedding()
    # -------------------------------------------------------------------------------------------------------

    # --------------------------------------Community Refiner------------------------------------------------
    refiner = Refiner(args, nx_graph, train_comms, valid_comms, embedding, train_pre_coms, valid_pre_coms)
    if not args.test:
        refiner.train()
    else:
        pred_coms = refiner.refine(test_pre_coms, test_comms)
        f1, nmi, jac = matrics(pred_coms, test_comms, num_node)
        print(f"f1-score:{f1}, nmi:{nmi}, jac:{jac}")
    # -------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cora',
                        help='Dataset name')
    parser.add_argument('--model_dir', type=str, default='ckpts',
                        help='Model directary')
    parser.add_argument('--train_size', type=int, default=7,
                        help='Train size')
    parser.add_argument('--valid_size', type=int, default=0,
                        help='Validation size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed')
    parser.add_argument('--with_feature', default=True,
                        help='With feature or without feature')
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Training device",)

    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha')
    parser.add_argument('--k', type=int, default=3,
                        help='K_hop subgraph')

    parser.add_argument('--encoder_lr', type=float, default=0.001,
                        help='Learning rate of the encoder')
    parser.add_argument('--encoder_epochs', type=int, default=100,
                        help='Number of epochs to encoder train')
    parser.add_argument('--triplet_samples', type=int, default=1000,
                        help='Number of samples per epoch')
    parser.add_argument('--encoder_batch_size', type=int, default=64,
                        help='Batch size')

    parser.add_argument('--refiner_epochs', type=int, default=10,
                        help='Number of epochs to modifier train.')
    parser.add_argument('--refiner_episode', type=int, default=2000,
                        help='Number of episode to modifier train')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="gamma")
    parser.add_argument("--lmbda", type=float, default=0.95,
                        help="lmbda")
    parser.add_argument("--eps", type=float, default=0.2,
                        help="Range of truncation parameters")
    parser.add_argument("--policy_lr", type=float, default=0.0001,
                        help="PolicyNet learning rate")
    parser.add_argument("--value_lr", type=float, default=0.001,
                        help="ValueNet learning rate")
    parser.add_argument("--max_step", type=int, default=15,
                        help="Max step of append or remove for train")
    parser.add_argument("--max_append_step", type=int, default=10,
                        help="Max step of append for valid and test")
    parser.add_argument("--max_remove_step", type=int, default=0,
                        help="Max step of remove for valid and test")

    #test
    parser.add_argument("--test", type=bool, default=True,
                        help="train, valid or test")

    args = parser.parse_args()

    main(args)