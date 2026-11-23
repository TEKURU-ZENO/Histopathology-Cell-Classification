import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, SAGEConv

class MLPHead(nn.Module):
    def __init__(self, in_features=512, hidden_features=128, num_classes=3, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)

class GATv2Head(nn.Module):
    def __init__(self, in_features=512, hidden_heads=4, hidden_per_head=16, 
                 out_features=32, num_classes=3, edge_dim=3, dropout=0.25):
        super().__init__()
        # Layer 1: 512 -> 16 * 4 = 64
        self.conv1 = GATv2Conv(in_features, hidden_per_head, heads=hidden_heads, 
                               concat=True, edge_dim=edge_dim, add_self_loops=True, fill_value="mean")
        self.act1 = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        # Layer 2: 64 -> 32, 1 head
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
        self.conv1 = SAGEConv(in_features, hidden_features, aggr='mean')
        self.act1 = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = SAGEConv(hidden_features, out_features, aggr='mean')
        self.classifier = nn.Linear(out_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return self.classifier(x)

mlp = MLPHead()
gat = GATv2Head()
sage = SAGEHead()

print(f"MLP total params:    {sum(p.numel() for p in mlp.parameters()):,}")
print(f"GATv2 total params:  {sum(p.numel() for p in gat.parameters()):,}")
print(f"  conv1: {sum(p.numel() for p in gat.conv1.parameters()):,}")
print(f"  conv2: {sum(p.numel() for p in gat.conv2.parameters()):,}")
print(f"  linear: {sum(p.numel() for p in gat.classifier.parameters()):,}")
print(f"SAGE total params:   {sum(p.numel() for p in sage.parameters()):,}")
print(f"  conv1: {sum(p.numel() for p in sage.conv1.parameters()):,}")
print(f"  conv2: {sum(p.numel() for p in sage.conv2.parameters()):,}")
print(f"  linear: {sum(p.numel() for p in sage.classifier.parameters()):,}")
