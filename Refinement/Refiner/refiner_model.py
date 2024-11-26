import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.helps import generate_outer_boundary, generate_inner_boundary
from Utils.metrics import ARI_score
from torch_geometric.nn import GCNConv

class MLP(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_channels)
        self.layer_norm1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_channels)
        self.layer_norm2 = nn.LayerNorm(output_channels)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

class GNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class Agent:
    def __init__(self, args, nx_graph, embedding, type):
        self.args = args
        self.nx_graph = nx_graph
        self.embedding = embedding
        self.type = type
        self.policyNet, self.valueNet = self.network_init()
        self.policyNet.to(self.args.device)
        self.valueNet.to(self.args.device)
        self.policyNet_opt = torch.optim.Adam(self.policyNet.parameters(), lr = self.args.policy_lr)
        self.valueNet_opt = torch.optim.Adam(self.valueNet.parameters(), lr=self.args.value_lr)
        self.action_space = []
        self.mask = None

    def network_init(self):
        policy = MLP(128, 64, 1)
        value = MLP(64, 32, 1)
        return policy, value

    def set_action(self, action_space):
        self.action_space = sorted(action_space)

    def set_mask(self, len):
        self.mask = torch.ones(len, dtype=torch.int)
        self.mask = self.mask.to(self.args.device)

    def take_action(self, state, nodefeats, mapping):
        index = sorted([idx for idx in mapping.keys() if mapping[idx] in self.action_space])

        com_idx = sorted([idx for idx in mapping.keys() if mapping[idx] in state])
        com_emb = torch.mean(nodefeats[com_idx, :], dim=0)

        s = nodefeats[index, :]
        com_emb_expanded = com_emb.unsqueeze(0).repeat(s.size(0), 1)
        s = torch.cat((s, com_emb_expanded), dim=1)

        tmp_mapping = {idx: mapping[idx_val] for idx, idx_val in enumerate(index)}

        embedding = self.policyNet(s)

        probs = F.softmax(torch.squeeze(embedding), dim=-1)

        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        masked_probs = probs * self.mask
        masked_probs /= masked_probs.sum()

        action_list = torch.distributions.Categorical(masked_probs)
        action = action_list.sample()

        return tmp_mapping[action.item()]

    def step(self, state, action, train_com):
        action_idx = self.action_space.index(action)
        self.mask[action_idx] = 0

        com = copy.deepcopy(state)

        if action in train_com:
            r_l = 1
        else:
            r_l = -1

        action_list = [self.action_space[i] for i in range(len(self.action_space)) if self.mask[i] == 1]

        if action == self.nx_graph.number_of_nodes():
            if set(action_list).isdisjoint(set(train_com)):
                reward = 2
            else:
                reward = -2
        elif action == self.nx_graph.number_of_nodes() + 1:
            if set(action_list).issubset(set(train_com)):
                reward = 2
            else:
                reward = -2
        else:
            if self.type == "APPEND":
                com.append(action)
                reward = ARI_score(list(self.nx_graph.nodes()), train_com, com) - ARI_score(list(self.nx_graph.nodes()), train_com, state) + r_l
                reward = (action - min(self.action_space)) / (max(self.action_space) - min(self.action_space)) + r_l
            else:
                com.remove(action)
                reward = ARI_score(list(self.nx_graph.nodes()), train_com, com) - ARI_score(list(self.nx_graph.nodes()),train_com, state) - r_l
                reward = (action - min(self.action_space)) / (max(self.action_space) - min(self.action_space)) - r_l

        return reward, com

    def get_prob(self, state, action, mapping, nodefeats):
        index = sorted([idx for idx in mapping.keys() if mapping[idx] in self.action_space])

        com_idx = sorted([idx for idx in mapping.keys() if mapping[idx] in state])
        com_emb = torch.mean(nodefeats[com_idx, :], dim=0)

        s = nodefeats[index, :]

        com_emb_expanded = com_emb.unsqueeze(0).repeat(s.size(0), 1)
        s = torch.cat((s, com_emb_expanded), dim=1)

        tmp_mapping = {idx: mapping[idx_val] for idx, idx_val in enumerate(index)}

        embedding = self.policyNet(s)

        probs = F.softmax(torch.squeeze(embedding), dim=-1)

        keys = list(tmp_mapping.keys())
        values = list(tmp_mapping.values())
        position = values.index(action)
        idx = keys[position]

        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        action_list = torch.distributions.Categorical(probs)

        action_prob = action_list.probs[idx]

        return action_prob.unsqueeze(0)

    def learn(self, transition_dict, mapping, train_pre_com, train_com):
        if transition_dict['states'] == []:
            return
        states = [state for state in transition_dict['states']]
        next_states = [next_state for next_state in transition_dict['next_states']]
        actions = transition_dict['actions']
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.args.device).view(-1,1)
        advantage = (rewards - torch.mean(rewards)) / torch.std(rewards)
        # print(advantage)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.args.device).view(-1,1)

        # next_q_target = []
        # for next_state in next_states:
        #     next_q_target.append(torch.mean(self.valueNet(next_state)))
        # next_q_target = torch.tensor(next_q_target).unsqueeze(1).to(self.args.device)
        #
        # td_target = rewards + self.args.gamma * next_q_target * (1 - dones)
        #
        # td_value = []
        # for state in states:
        #     td_value.append(torch.mean(self.valueNet(state)))
        # td_value = torch.tensor(td_value).unsqueeze(1).to(self.args.device)
        #
        # td_delta = td_target - td_value
        #
        # td_delta = td_delta.cpu().detach().numpy()
        #
        # advantage = 0
        # advantage_list = []
        #
        # for delta in td_delta[::-1]:
        #     advantage = self.args.gamma * self.args.lmbda * advantage + delta
        #     advantage_list.append(advantage)
        #
        # advantage_list.reverse()
        # advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.args.device)

        old_log_probs = []
        comm = copy.deepcopy(train_pre_com)
        for i in range(len(actions)):
            action = actions[i]
            prob = self.get_prob(comm, action, mapping, transition_dict['states'][i])
            old_log_probs.append(torch.log(prob))
            if action < self.nx_graph.number_of_nodes():
                if self.type == "APPEND":
                    comm.append(action)
                else:
                    comm.remove(action)

        old_log_probs = torch.cat(old_log_probs).view(-1,1).detach()
        # print(old_log_probs)

        for _ in range(self.args.refiner_epochs):
            log_probs = []
            comm = copy.deepcopy(train_pre_com)
            for i in range(len(actions)):
                action = actions[i]
                prob = self.get_prob(comm, action, mapping, transition_dict['states'][i])
                log_probs.append(torch.log(prob))
                if action < self.nx_graph.number_of_nodes():
                    if self.type == "APPEND":
                        comm.append(action)
                    else:
                        comm.remove(action)

            log_probs = torch.cat(log_probs).view(-1,1)

            # print(actions)
            # print("log_probs=",log_probs)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage

            surr2 = torch.clamp(ratio, 1 - self.args.eps, 1 + self.args.eps) * advantage

            policy_loss = torch.mean(-torch.min(surr1, surr2))
            # print("policy_loss=",policy_loss)
            # new_td_value = []
            # for state in states:
            #     new_td_value.append(torch.mean(self.valueNet(state)).unsqueeze(0))
            # new_td_value = torch.cat(new_td_value).view(-1,1)

            # value_loss = torch.mean(F.mse_loss(new_td_value, td_target.detach()))

            self.policyNet_opt.zero_grad()
            # self.valueNet_opt.zero_grad()

            policy_loss.backward()
            # value_loss.backward()

            self.policyNet_opt.step()
            # self.valueNet_opt.step()

    def take_action_greedy(self, state, nodefeats, mapping):
        index = sorted([idx for idx in mapping.keys() if mapping[idx] in self.action_space])

        com_idx = sorted([idx for idx in mapping.keys() if mapping[idx] in state])
        com_emb = torch.mean(nodefeats[com_idx, :], dim=0)

        s = nodefeats[index, :]
        com_emb_expanded = com_emb.unsqueeze(0).repeat(s.size(0), 1)
        s = torch.cat((s, com_emb_expanded), dim=1)

        tmp_mapping = {idx: mapping[idx_val] for idx, idx_val in enumerate(index)}

        embedding = self.policyNet(s)

        probs = F.softmax(torch.squeeze(embedding), dim=-1)

        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        masked_probs = probs * self.mask
        masked_probs /= masked_probs.sum()

        action = torch.argmax(masked_probs)

        return tmp_mapping[action.item()]