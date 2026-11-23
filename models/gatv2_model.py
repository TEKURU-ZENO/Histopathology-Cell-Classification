import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, SAGEConv

class GATv2Head(nn.Module):
    def __init__(self, in_features=512, hidden_heads=4, hidden_per_head=16, 
                 out_features=32, num_classes=3, edge_dim=3, dropout=0.25):
        super().__init__()
        # Layer 1: 512 -> 16 * 4 = 64, edge_dim = 3
        self.conv1 = GATv2Conv(in_features, hidden_per_head, heads=hidden_heads, 
                               concat=True, edge_dim=edge_dim, add_self_loops=True, fill_value="mean")
        self.act1 = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        # Layer 2: 64 -> 32, 1 head, edge_dim = 3
        self.conv2 = GATv2Conv(hidden_heads * hidden_per_head, out_features, heads=1, 
                               concat=False, edge_dim=edge_dim, add_self_loops=True, fill_value="mean")
        self.classifier = nn.Linear(out_features, num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return self.classifier(x)

class SAGEHead(nn.Module):
    def __init__(self, in_features=512, hidden_features=64, 
                 out_features=32, num_classes=3, dropout=0.25):
        super().__init__()
        # Layer 1: 512 -> 64 (mean aggregation)
        self.conv1 = SAGEConv(in_features, hidden_features, aggr='mean')
        self.act1 = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        # Layer 2: 64 -> 32
        self.conv2 = SAGEConv(hidden_features, out_features, aggr='mean')
        self.classifier = nn.Linear(out_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return self.classifier(x)
