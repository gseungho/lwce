"""MLP backbone of the tabular and network domains."""

import torch.nn as nn


class MLP(nn.Module):
    """256-128-64 MLP: each hidden layer is Linear -> BatchNorm1d -> ReLU -> Dropout."""

    def __init__(self, input_dim, num_classes, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
