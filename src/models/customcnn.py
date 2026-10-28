import torch
from torch import nn

class FeatherCam(nn.Module):
    ...

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
