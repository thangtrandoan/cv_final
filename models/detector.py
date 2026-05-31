from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyGridDetector(nn.Module):
    def __init__(self, num_classes: int = 5, num_anchors: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_channels = num_anchors * (5 + num_classes)

        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512),
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 1024),
        )
        self.head = nn.Sequential(
            ConvBlock(1024, 512),
            nn.Conv2d(512, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
