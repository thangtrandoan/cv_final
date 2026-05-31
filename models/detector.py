from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


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


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes: tuple[int, ...] = (5, 9, 13)) -> None:
        super().__init__()
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2) for size in pool_sizes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, *[pool(x) for pool in self.pools]], dim=1)


class TinyGridDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        num_anchors: int = 3,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_channels = num_anchors * (5 + num_classes)
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.head = nn.Sequential(
            ConvBlock(2048, 1024, kernel_size=1),
            SpatialPyramidPooling(),
            ConvBlock(4096, 1024, kernel_size=1),
            ConvBlock(1024, 512),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(512, out_channels, kernel_size=1),
        )
        self._init_detection_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def _init_detection_head(self) -> None:
        final_conv = self.head[-1]
        if not isinstance(final_conv, nn.Conv2d) or final_conv.bias is None:
            return
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_conv.bias, 0.0)
        values_per_anchor = 5 + self.num_classes
        for anchor_idx in range(self.num_anchors):
            objectness_channel = anchor_idx * values_per_anchor + 4
            final_conv.bias.data[objectness_channel] = -4.0
