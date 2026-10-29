import torch
from torch import nn

import torch
import torch.nn as nn


class FeatherCam(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        # Stage 1: 3x224x224 -> 32x112x112 (2 Convs: 3->32, 32->32)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 32x112x112 -> 64x56x56 (2 Convs: 32->64, 64->64)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # Stage 3: 64x56x56 -> 128x28x28 (1 Conv: 64->128)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # Stage 4: 128x28x28 -> 256x14x14 (1 Conv: 128->256)
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MockFeatherCam(nn.Module):
    """
    Lightweight mock module for unit tests and factory validation.
    Accepts (B, C, H, W) and outputs (B, num_classes).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 200):
        super().__init__()
        self.num_classes = num_classes
        # Pools any spatial dimension (e.g., 224x224, 64x64) down to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C, 1, 1)
        x = self.pool(x)
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = torch.flatten(x, 1)
        # Output logits: (B, num_classes)
        return self.fc(x)
