import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, dropout_rate):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x