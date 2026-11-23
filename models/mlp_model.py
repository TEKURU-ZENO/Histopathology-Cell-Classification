import torch
import torch.nn as nn

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
