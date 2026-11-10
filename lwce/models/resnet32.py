"""ResNet-32 for CIFAR-10/100 (32 x 32 input).

Architecture (He et al. 2016, Table 6, CIFAR variant):
    conv1:  3x3, 16 channels, stride 1 (no max-pool)
    layer1: 6 x BasicBlock(16, 16, stride=1)
    layer2: 6 x BasicBlock(16, 32, stride=2)  (the first block downsamples)
    layer3: 6 x BasicBlock(32, 64, stride=2)  (the first block downsamples)
    avgpool: global average pool -> (B, 64)
    fc:     Linear(64, num_classes), or a cosine classifier for LDAM

Depth: 1 + 3 x (6 x 2) + 1 = 32 weight layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormedLinear(nn.Module):
    """Cosine classifier: L2-normalised features and weights, outputs in [-1, 1].

    LDAM (Cao et al. 2019) assumes this head and scales its outputs by s = 30; with a
    plain linear head the scaled logits explode, so it is used for LDAM only.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class BasicBlock(nn.Module):
    """Two-layer residual block (no bottleneck), as in ResNet-20/32/44/56 for CIFAR."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # 1x1 projection shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet32(nn.Module):
    """ResNet-32 for CIFAR-10/100."""

    def __init__(self, num_classes=10, normed_head=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, n_blocks=6, stride=1)
        self.layer2 = self._make_layer(16, 32, n_blocks=6, stride=2)
        self.layer3 = self._make_layer(32, 64, n_blocks=6, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Cosine classifier for LDAM (normed_head=True), a standard linear layer otherwise.
        self.fc = NormedLinear(64, num_classes) if normed_head else nn.Linear(64, num_classes)
        self._init_weights()

    def _make_layer(self, in_planes, planes, n_blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]   # the first block may downsample
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming-normal convolutions, BatchNorm weight 1 and bias 0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # (B, 16, 32, 32)
        out = self.layer1(out)                  # (B, 16, 32, 32)
        out = self.layer2(out)                  # (B, 32, 16, 16)
        out = self.layer3(out)                  # (B, 64,  8,  8)
        out = self.avgpool(out).flatten(1)      # (B, 64)
        return self.fc(out)                     # (B, num_classes)


def build_resnet32(num_classes=10, normed_head=False):
    """ResNet-32; normed_head=True gives the cosine classifier used with LDAM."""
    return ResNet32(num_classes=num_classes, normed_head=normed_head)
