"""PointNet model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).flatten()
        iden = iden.view(1, self.k*self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)

        return x

class PointNet(nn.Module):
    def __init__(self, n_classes=10, use_tnet=True):
        super(PointNet, self).__init__()
        self.n_classes = n_classes
        self.use_tnet = use_tnet

        if use_tnet:
            self.tnet1 = TNet(k=3)
            self.tnet2 = TNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, N, 3) point cloud
            return_features: if True, return intermediate features
        Returns:
            logits: (B, n_classes)
        """
        batch_size = x.size(0)
        n_points = x.size(1)

        # Transpose to (B, 3, N) for conv1d
        x = x.transpose(2, 1)

        # Input transform
        if self.use_tnet:
            trans = self.tnet1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)

        # MLP (64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        if self.use_tnet:
            trans_feat = self.tnet2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        point_features = x  # (B, 64, N)

        # MLP (64, 128, 1024)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        # Global pooling
        global_feature = torch.max(x, 2, keepdim=False)[0]  # (B, 1024)

        # Classification head
        x = F.relu(self.bn6(self.fc1(global_feature)))
        x = F.relu(self.bn7(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        if return_features:
            return x, global_feature, point_features
        return x

# Create model
n_classes = len(train_dataset.classes)
model = PointNet(n_classes=n_classes, use_tnet=True).to(device)
