import torch
from torch import nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ResNeXtStageBlock(nn.Module):
    def __init__(self, in_c, c_per_group, groups, out_c, stride=1):
        super().__init__()
        mid_c = c_per_group * groups

        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)

        # Downsampling stride applied at the 3x3 grouped convolution
        self.conv2 = nn.Conv2d(
            mid_c, mid_c, kernel_size=3, stride=stride, padding=1,
            groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_c)

        self.conv3 = nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.act = nn.GELU()

        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + res)


class FeatherNext(nn.Module):
    def __init__(self, num_classes=200, groups=8):
        super().__init__()
        # Stem: Keep 224x224 intact for fine texture capture
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # Stage configs: (c_per_group, out_channels, num_blocks, stride_for_block0)
        stage_specs = [
            (4, 64, 2, 1),      # Stage 1: 224x224
            (8, 128, 3, 2),     # Stage 2: 112x112
            (16, 256, 4, 2),    # Stage 3: 56x56
            (32, 512, 6, 2),    # Stage 4: 28x28
            (64, 1024, 4, 2),   # Stage 5: 14x14
            (128, 2048, 3, 2),  # Stage 6: 7x7
        ]

        self.stages = nn.ModuleList()
        in_c = 64
        for c_group, out_c, num_blocks, stride in stage_specs:
            layers = []
            for b_idx in range(num_blocks):
                s = stride if b_idx == 0 else 1
                curr_in = in_c if b_idx == 0 else out_c
                layers.append(ResNeXtStageBlock(curr_in, c_group, groups, out_c, stride=s))
            self.stages.append(nn.Sequential(*layers))
            in_c = out_c

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

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
