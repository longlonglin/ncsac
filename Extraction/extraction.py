import copy
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from Utils.helps import choose_seed


class Extraction:
    def __init__(self, args, nx_graph, nodefeats, communities):
        self.args = args
        self.nx_graph = nx_graph
        self.nodefeats = nodefeats
        self.communities = communities
        self.degree, self.att_degree = self.prepare()

    def prepare(self):
        degree_dict = dict(self.nx_graph.degree())
        all_nodes = sorted(self.nx_graph.nodes())
        degree = np.array([degree_dict[node] for node in all_nodes])

        att_degree_vector = self.nodefeats.sum(axis=0) - 1
        att_degree = self.nodefeats.dot(att_degree_vector.T)

        return degree, att_degree

    # def return_pre_community(self, seed):
    #     pre_com = {seed}
    #     total_degree = np.sum(self.degree)
    #     total_att_degree = np.sum(self.att_degree)
    #     P = {seed}
    #     all_nodes = set(self.nx_graph.nodes)
    #     min_conductance = 1
    #     for i in range(self.args.k):
    #         for v in P:
    #             temp = list(self.nx_graph.neighbors(v))
    #             P = P | set(temp)
    #
    #         rest = all_nodes - P
    #
    #         edges_between_sets = self.nx_graph.edges(P, data=False)
    #         top_cut = sum(1 for u, v in edges_between_sets if u in P and v in rest)
    #
    #         c_degree = sum(dict(self.nx_graph.degree(P)).values())
    #
    #         top_con = top_cut / min(c_degree, total_degree - c_degree)
    #
    #         att_cut = (self.nodefeats[list(P)].dot(self.nodefeats[list(rest)].T)).sum()
    #
    #         c_att_degree = (self.att_degree[list(P)]).sum()
    #
    #         att_con = att_cut / min(c_att_degree, total_att_degree - c_att_degree)
    #
    #         conductance = self.args.alpha * top_con + (1 - self.args.alpha) * att_con
    #
    #         if i == 0:
    #             min_conductance = conductance
    #             pre_com = pre_com | P
    #         else:
    #             if conductance < min_conductance:
    #                 min_conductance = conductance
    #                 pre_com = pre_com | P
    #
    #     return pre_com


    def return_pre_community(self, seed):
        pre_com = {seed}

        top_cut = self.degree[seed]
        c_degree = self.degree[seed]
        total_degree = np.sum(self.degree)

        att_cut = self.att_degree[seed]
        c_att_degree = self.att_degree[seed]
        total_att_degree = np.sum(self.att_degree)

        att = copy.deepcopy(self.nodefeats[seed, :])

        P, Q, C = {seed}, {seed}, {seed}
        min_conductance = 1
        for i in range(self.args.k):
            for v in P:
                temp = list(self.nx_graph.neighbors(v))
                P = P | set(temp)

            Q = P - Q

            for v in Q:
                top_cut = top_cut + self.degree[v] - 2 * len(C & set(list(self.nx_graph.neighbors(v))))
                c_degree += self.degree[v]

                v_vector = self.nodefeats[v, :]
                att_cut = att_cut + self.att_degree[v] - 2 * v_vector.dot(att.T)
                c_att_degree += self.att_degree[v]

                att += v_vector

                C.add(v)

            top_con = top_cut / min(c_degree, total_degree - c_degree)

            att_con = att_cut / min(c_att_degree, total_att_degree - c_att_degree)

            Q = P

            conductance = self.args.alpha * top_con + (1 - self.args.alpha) * att_con

            if i == 0:
                min_conductance = conductance
                pre_com = pre_com | P
            else:
                if conductance < min_conductance:
                    min_conductance = conductance
                    pre_com = pre_com | P

        return pre_com

    def extract(self):
        print("Start extracting preliminary community...")

        train_pre_coms = []

        start_time = time.time()

        for com in self.communities:
            seed = choose_seed(com, self.nx_graph)
            train_pre_coms.append(list(self.return_pre_community(seed)))

        end_time = time.time()

        print("time=", end_time - start_time)

        print("")

        return train_pre_coms
