import torch
import torch.optim as optim
import random
import torch.nn.functional as F

from tqdm import tqdm
from .gnn import GNN
from Utils.helps import choose_seed


class Encoder:
    def __init__(self, args, nx_graph, graph_data, train_comms):
        self.args = args
        self.nx_graph = nx_graph
        self.graph_data = graph_data
        self.train_comms = train_comms

        self.num_node, input_dim = graph_data.x.size(0), graph_data.x.size(1)

        self.gnn_encoder = GNN(input_dim, 128, 64, 0.5)
        self.gnn_encoder.to(self.args.device)
        self.graph_data.to(self.args.device)

    def get_embedding(self):
        with torch.no_grad():
            embedding =  self.gnn_encoder(self.graph_data.x, self.graph_data.edge_index)

        zeros = torch.randn(2, 64, device=self.args.device)
        embedding = torch.cat((embedding, zeros), dim=0)

        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def dis(self, A, B):
        cosine_similarity = F.cosine_similarity(A.unsqueeze(0), B, dim=1)
        cosine_distance = 1 - cosine_similarity

        return cosine_distance

    def tripletLoss(self, anchor, positive, negative, margin):
        loss = torch.clamp(self.dis(anchor, positive) - self.dis(anchor, negative) + margin, min=0.0)

        return loss.mean()

    def train(self):
        optimizer = optim.Adam(self.gnn_encoder.parameters(), lr=self.args.encoder_lr)

        print("Start training gnn encoder...")

        for epoch in range(self.args.encoder_epochs):
            self.gnn_encoder.train()

            train_com = random.choice(self.train_comms)
            seed = choose_seed(train_com, self.nx_graph)

            positive_samples = [node for node in train_com if node != seed]
            negative_samples = list(set(self.nx_graph.nodes()) - set(train_com))

            positive_data = [random.choice(positive_samples) for i in range(self.args.triplet_samples)]
            negative_data = [random.choice(negative_samples) for i in range(self.args.triplet_samples)]

            running_loss = 0.0

            with tqdm(total=self.args.triplet_samples // self.args.encoder_batch_size, desc=f"Epoch {epoch + 1}/{self.args.encoder_epochs}", ncols=100) as pbar:
                for i in range(self.args.triplet_samples // self.args.encoder_batch_size):
                    optimizer.zero_grad()
                    embeddings = self.gnn_encoder(self.graph_data.x, self.graph_data.edge_index)

                    current_batch_size = min(self.args.encoder_batch_size, self.args.triplet_samples - (i + 1) * self.args.encoder_batch_size)
                    indices = random.sample(range(self.args.triplet_samples), current_batch_size)

                    positive_batch_data = [positive_data[i] for i in indices]
                    negative_batch_data = [negative_data[i] for i in indices]

                    loss_t = self.tripletLoss(embeddings[seed], embeddings[positive_batch_data], embeddings[negative_batch_data], margin=1.0)

                    loss_t.backward()
                    optimizer.step()

                    running_loss += loss_t.item()

                    pbar.set_postfix({'loss': running_loss / (i + 1)})
                    pbar.update(1)

        print("")