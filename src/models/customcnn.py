import torch
import torch.nn as nn


class FeatherCam(nn.Module):
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.num_classes = num_classes

        # Stage 1: 3x224x224 -> 32x112x112 (2 Convs: 3->32, 32->32)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 2: 32x112x112 -> 64x56x56 (2 Convs: 32->64, 64->64)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 3: 64x56x56 -> 128x28x28 (1 Conv: 64->128)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 4: 128x28x28 -> 256x14x14 (1 Conv: 128->256; post-ReLU at stage4[2] is 28x28)
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(256, num_classes)

    def extract_feature_maps(self, x: torch.Tensor):
        """Return multi-scale spatial feature maps (f1, f2, f3, f4) for Grad-CAM / Phase 2 decoders."""
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return f1, f2, f3, f4

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-dim pooled latent embedding prior to classification head."""
        _, _, _, f4 = self.extract_feature_maps(x)
        return torch.flatten(self.gap(f4), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        latent = self.dropout(latent)
        return self.fc(latent)


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
