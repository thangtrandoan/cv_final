from __future__ import annotations

import random
from typing import Any

import torch
from PIL import Image, ImageEnhance


class DetectionTransform:
    def __init__(
        self,
        img_size: int = 416,
        train: bool = True,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.15,
    ) -> None:
        self.img_size = img_size
        self.train = train
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter

    def __call__(self, image: Image.Image, targets: list[dict[str, Any]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        image = image.convert("RGB")
        width, height = image.size
        targets = [{"class_id": item["class_id"], "bbox": list(item["bbox"])} for item in targets]

        if self.train and random.random() < self.hflip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for item in targets:
                xmin, ymin, xmax, ymax = item["bbox"]
                item["bbox"] = [width - xmax, ymin, width - xmin, ymax]

        if self.train and self.color_jitter > 0:
            image = self._color_jitter(image)

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        scale_x = self.img_size / width
        scale_y = self.img_size / height
        for item in targets:
            xmin, ymin, xmax, ymax = item["bbox"]
            item["bbox"] = [
                max(0.0, min(self.img_size, xmin * scale_x)),
                max(0.0, min(self.img_size, ymin * scale_y)),
                max(0.0, min(self.img_size, xmax * scale_x)),
                max(0.0, min(self.img_size, ymax * scale_y)),
            ]

        tensor = self._to_tensor(image)
        return tensor, targets

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            image = enhancer_cls(image).enhance(factor)
        return image

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        data = data.view(image.size[1], image.size[0], 3)
        return data.permute(2, 0, 1).float().div(255.0)
