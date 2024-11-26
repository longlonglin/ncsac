import copy
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from Refinement.Refiner.refiner_model import Agent, GNN
from Utils.helps import generate_outer_boundary, generate_inner_boundary
from Utils.metrics import matrics


class Refiner:
    def __init__(self, args, nx_graph, train_comms, valid_comms, embedding, train_pre_coms, valid_pre_coms):
        self.args = args
        self.nx_graph = nx_graph
        self.train_comms = train_comms
        self.valid_comms = valid_comms
        self.embedding = embedding
        self.train_pre_coms = train_pre_coms
        self.valid_pre_coms = valid_pre_coms
        self.append_agent = Agent(args, nx_graph, embedding, type = 'APPEND')
        self.remove_agent = Agent(args, nx_graph, embedding, type = 'REMOVE')
        self.gnn = GNN(64, 32,64)
        self.gnn.to(args.device)

    def load_net(self, file_name, type):
        print(f"Load net from {file_name}")
        data = torch.load(file_name)
        if type == "APPEND":
            self.append_agent.policyNet.load_state_dict(data)
        else:
            self.remove_agent.policyNet.load_state_dict(data)

    def save_net(self, file_name, type):
        print(f"Save net in {file_name}")
        if type == "APPEND":
            data = self.append_agent.policyNet.state_dict()
        else:
            data = self.remove_agent.policyNet.state_dict()

        torch.save(data, file_name)
        print("")

    def generate_position_flag(self, embedding, mapping, state):
        result = torch.zeros((embedding.shape[0], 1))
        for idx, node in mapping.items():
            if node in state:
                result[idx, 0] = 1
        return result

    def update(self, state, mapping, embedding):
        edges = list(self.nx_graph.subgraph(state).edges())
        e_tensor = torch.zeros((2, len(edges)), dtype=int).to(self.args.device)
        revert_mapping = {node: idx for idx, node in mapping.items()}

        for i in range(len(edges)):
            u, v = edges[i]
            e_tensor[0][i], e_tensor[1][i] = revert_mapping[u], revert_mapping[v]

        new_e = self.gnn(embedding, e_tensor).detach()

        # position_flag = self.generate_position_flag(new_e, mapping, state).to(self.args.device)
        # new_e = torch.cat((position_flag, new_e.detach()), dim=1)

        return new_e

    def train(self):

        append_return_list, remove_return_list = [], []
        best_f, best_n, best_j = 0, 0, 0

        print("Start training appendNet...")

        for i in range(20):
            with tqdm(total=int(self.args.refiner_episode / 20), desc='Iteration %d' % (i+1)) as pbar:
                for episode in range(int(self.args.refiner_episode / 20)):
                    random_index = random.randint(0, len(self.train_comms) - 1)
                    train_com = self.train_comms[random_index]
                    # print(train_com)
                    train_pre_com = self.train_pre_coms[random_index]
                    # print(train_pre_com)

                    boundary = generate_outer_boundary(self.nx_graph, train_pre_com, train_com)
                    boundary.append(self.nx_graph.number_of_nodes())
                    self.append_agent.set_action(boundary)
                    # print(self.append_agent.action_space)
                    action_len = len(self.append_agent.action_space)

                    if action_len <= 1:
                        done = True
                    else:
                        done = False
                    self.append_agent.set_mask(action_len)

                    all_nodes = sorted(train_pre_com + boundary + [self.nx_graph.number_of_nodes() + 1])
                    embedding = self.embedding[all_nodes, :]
                    mapping = {idx: node for idx, node in enumerate(all_nodes)}
                    # print(mapping)
                    # position_flag = self.generate_position_flag(embedding, mapping, train_pre_com).to(self.args.device)
                    # embedding = torch.cat((position_flag, embedding), dim=1)

                    state = copy.deepcopy(train_pre_com)

                    episode_return = 0

                    transition_dict = {
                        'states': [],
                        'actions': [],
                        'next_states': [],
                        'rewards': [],
                        'dones': [],
                    }

                    step = 0

                    while not done:
                        transition_dict['states'].append(embedding)

                        action = self.append_agent.take_action(state, embedding, mapping)
                        reward, next_state = self.append_agent.step(state, action, train_com)

                        step += 1

                        transition_dict['actions'].append(action)
                        transition_dict['rewards'].append(reward)

                        if step == action_len or (step >= self.args.max_step):
                            done = True

                        transition_dict['dones'].append(done)

                        state = next_state

                        embedding = self.update(state, mapping, embedding)
                        transition_dict['next_states'].append(embedding)

                        episode_return += reward
                    # print(transition_dict)
                    append_return_list.append(episode_return)
                    self.append_agent.learn(transition_dict, mapping, train_pre_com ,train_com)

                    if (episode + 1) % 5 == 0:
                        pbar.set_postfix({'episode': '%d' % (self.args.refiner_episode / 20 * i + episode + 1),
                                          'return': '%.3f' % np.mean(append_return_list[-5:])})
                    pbar.update(1)

            f, n, j = matrics(self.valid_pre_coms, self.valid_comms, self.nx_graph.number_of_nodes())
            valid_preds = self.refine(self.valid_pre_coms, self.valid_comms, r=False, test=False)
            new_f, new_n, new_j = matrics(valid_preds, self.valid_comms, self.nx_graph.number_of_nodes())
            if new_f - f > best_f:
                best_f = new_f - f
                self.save_net(self.args.model_dir + "/" + self.args.dataset + f"/com_append_episode" + str(int(self.args.refiner_episode / 20 * i + episode + 1)) + ".pt", type="APPEND")
        self.save_net(self.args.model_dir + "/" + self.args.dataset + f"/com_append.pt", type="APPEND")

        print("")

        print("Start training removeNet...")

        best_f, best_n, best_j = 0, 0, 0

        for i in range(20):
            with tqdm(total=int(self.args.refiner_episode / 20), desc='Iteration %d' % (i+1)) as pbar:
                for episode in range(int(self.args.refiner_episode / 20)):
                    random_index = random.randint(0, len(self.train_comms) - 1)
                    train_com = self.train_comms[random_index]

                    train_pre_com = self.train_pre_coms[random_index]

                    boundary = generate_outer_boundary(self.nx_graph, train_pre_com, train_com)
                    com = generate_inner_boundary(self.nx_graph, train_pre_com, train_com)
                    com.append(self.nx_graph.number_of_nodes() + 1)
                    self.remove_agent.set_action(com)

                    action_len = len(self.remove_agent.action_space)
                    self.remove_agent.set_mask(action_len)
                    # print(self.remove_agent.action_space)
                    all_nodes = sorted(train_pre_com + boundary + [self.nx_graph.number_of_nodes(), self.nx_graph.number_of_nodes() + 1])
                    embedding = self.embedding[all_nodes, :]
                    mapping = {idx: node for idx, node in enumerate(all_nodes)}
                    # position_flag = self.generate_position_flag(embedding, mapping, train_pre_com).to(self.args.device)
                    # embedding = torch.cat((position_flag, embedding), dim=1)

                    state = copy.deepcopy(train_pre_com)
                    done = False
                    episode_return = 0

                    transition_dict = {
                        'states': [],
                        'actions': [],
                        'next_states': [],
                        'rewards': [],
                        'dones': [],
                    }

                    step = 0

                    while not done:
                        transition_dict['states'].append(embedding)

                        action = self.remove_agent.take_action(state, embedding, mapping)
                        reward, next_state = self.remove_agent.step(state, action, train_com)

                        step += 1

                        transition_dict['actions'].append(action)
                        transition_dict['rewards'].append(reward)

                        if step == action_len - 1 or step >= self.args.max_step:
                            done = True

                        transition_dict['dones'].append(done)

                        state = next_state

                        embedding = self.update(state, mapping, embedding)
                        transition_dict['next_states'].append(embedding)

                        episode_return += reward

                    remove_return_list.append(episode_return)
                    # print(transition_dict)
                    self.remove_agent.learn(transition_dict, mapping, train_pre_com, train_com)

                    if (episode + 1) % 5 == 0:
                        pbar.set_postfix({'episode': '%d' % (self.args.refiner_episode / 20 * i + episode + 1),
                                          'return': '%.3f' % np.mean(remove_return_list[-5:])})
                    pbar.update(1)

            f, n, j = matrics(self.valid_pre_coms, self.valid_comms, self.nx_graph.number_of_nodes())
            valid_preds = self.refine(self.valid_pre_coms, self.valid_comms, a=False, test=False)
            new_f, new_n, new_j = matrics(valid_preds, self.valid_comms, self.nx_graph.number_of_nodes())
            if new_f - f > best_f:
                best_f = new_f - f
                self.save_net(self.args.model_dir + "/" + self.args.dataset + f"/com_remove_episode" + str(int(self.args.refiner_episode / 20 * i + episode + 1)) + ".pt", type = "REMOVE")
        self.save_net(self.args.model_dir + "/" + self.args.dataset + f"/com_remove.pt", type="REMOVE")
        print("")

    def refine(self, pre_comms, comms, a=True, r=True, test=True):
        if test:
            self.load_net(self.args.model_dir + "/" + self.args.dataset + "/com_append_episode800.pt", "APPEND")
            self.load_net(self.args.model_dir + "/" + self.args.dataset + "/com_remove.pt","REMOVE")

        pred_com = []

        start_time = time.time()
        for i in range(len(comms)):
            append, remove = a, r

            state = copy.deepcopy(pre_comms[i])
            append_step, remove_step = 0, 0

            boundary = generate_outer_boundary(self.nx_graph, pre_comms[i], comms[i])
            boundary.append(self.nx_graph.number_of_nodes())

            self.append_agent.set_action(boundary)
            self.append_agent.set_mask(len(self.append_agent.action_space))

            r_com = generate_inner_boundary(self.nx_graph, pre_comms[i], comms[i])
            r_com.append(self.nx_graph.number_of_nodes() + 1)
            self.remove_agent.set_action(r_com)
            self.remove_agent.set_mask(len(self.remove_agent.action_space))

            all_nodes = sorted(pre_comms[i] + boundary + [self.nx_graph.number_of_nodes() + 1])
            embedding = self.embedding[all_nodes, :]
            mapping = {idx: node for idx, node in enumerate(all_nodes)}
            # position_flag = self.generate_position_flag(embedding, mapping, com).to(self.args.device)
            # embedding = torch.cat((position_flag, embedding), dim=1)

            while append or remove:
                if append:
                    append_action = self.append_agent.take_action_greedy(state, embedding, mapping)
                    # print(append_action)
                    if append_action == self.nx_graph.number_of_nodes() or append_step >= self.args.max_append_step:
                        append = False
                    else:
                        action_idx = self.append_agent.action_space.index(append_action)
                        self.append_agent.mask[action_idx] = 0
                        state.append(append_action)
                        append_step += 1

                if remove:
                    remove_action = self.remove_agent.take_action_greedy(state, embedding, mapping)
                    # print(remove_action)
                    if remove_action == self.nx_graph.number_of_nodes() + 1 or remove_step >= self.args.max_remove_step or remove_step == len(self.remove_agent.action_space) - 1:
                        remove = False
                    else:
                        action_idx = self.remove_agent.action_space.index(remove_action)
                        self.remove_agent.mask[action_idx] = 0
                        state.remove(remove_action)
                        remove_step += 1

                embedding = self.update(state, mapping, embedding)

            pred_com.append(state)
        end_time = time.time()
        # print("time=", end_time - start_time)

        return pred_com