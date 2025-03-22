import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, time_embedding_dim=128):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128 + time_embedding_dim, num_classes)
        self.time_embedding = nn.Linear(1, time_embedding_dim)

    def forward(self, x, t):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        t = t.float().unsqueeze(1)
        t_emb = F.relu(self.time_embedding(t))
        x = torch.cat([x, t_emb], dim=1)
        x = self.fc2(x)
        return x
