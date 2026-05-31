from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FCOSHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 5) -> None:
        super().__init__()
        self.cls_tower = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, in_channels),
        )
        self.reg_tower = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, in_channels),
        )
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.cnt_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_features = self.cls_tower(x)
        reg_features = self.reg_tower(x)
        cls_logits = self.cls_head(cls_features)
        reg_preds = F.softplus(self.reg_head(reg_features))
        cnt_logits = self.cnt_head(reg_features)
        return cls_logits, reg_preds, cnt_logits

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)


class TinyGridDetector(nn.Module):
    level_strides = {"p3": 8, "p4": 16, "p5": 32}

    def __init__(self, num_classes: int = 5, num_anchors: int = 3, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.p5_conv = nn.Conv2d(2048, 256, kernel_size=1)
        self.p4_conv = nn.Conv2d(1024, 256, kernel_size=1)
        self.p3_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.p5_smooth = ConvBlock(256, 256)
        self.p4_smooth = ConvBlock(256, 256)
        self.p3_smooth = ConvBlock(256, 256)
        self.head = FCOSHead(256, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.p5_conv(c5)
        p4 = self.p4_conv(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.p3_conv(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p5 = self.p5_smooth(p5)
        p4 = self.p4_smooth(p4)
        p3 = self.p3_smooth(p3)

        return {
            "p3": self.head(p3),
            "p4": self.head(p4),
            "p5": self.head(p5),
        }
