from __future__ import annotations

import json
import random
from pathlib import Path

import torch


DEFAULT_ANCHORS = torch.tensor(
    [
        [0.08, 0.12],
        [0.18, 0.25],
        [0.35, 0.45],
    ],
    dtype=torch.float32,
)


def anchor_iou_wh(box_wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    inter = torch.minimum(box_wh[..., None, :], anchors).prod(dim=-1)
    box_area = box_wh.prod(dim=-1, keepdim=True)
    anchor_area = anchors.prod(dim=-1)
    return inter / (box_area + anchor_area - inter).clamp(min=1e-6)


def load_box_wh(annotation_path: str | Path) -> torch.Tensor:
    annotation_path = Path(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    image_info = {item["id"]: item for item in data["images"]}
    wh_values: list[list[float]] = []
    for ann in data["annotations"]:
        info = image_info[ann["image_id"]]
        xmin, ymin, xmax, ymax = ann["bbox"]
        w = max(1e-6, (xmax - xmin) / info["width"])
        h = max(1e-6, (ymax - ymin) / info["height"])
        wh_values.append([w, h])
    return torch.tensor(wh_values, dtype=torch.float32)


def kmeans_anchors(annotation_path: str | Path, k: int = 3, iterations: int = 50, seed: int = 42) -> torch.Tensor:
    boxes = load_box_wh(annotation_path)
    if boxes.shape[0] < k:
        return DEFAULT_ANCHORS[:k].clone()

    random.seed(seed)
    indices = random.sample(range(boxes.shape[0]), k)
    centers = boxes[indices].clone()

    for _ in range(iterations):
        distances = 1.0 - anchor_iou_wh(boxes, centers)
        labels = distances.argmin(dim=1)
        new_centers = centers.clone()
        for idx in range(k):
            assigned = boxes[labels == idx]
            if assigned.numel() > 0:
                new_centers[idx] = assigned.median(dim=0).values
        if torch.allclose(new_centers, centers, atol=1e-5):
            break
        centers = new_centers

    areas = centers.prod(dim=1)
    return centers[areas.argsort()].clamp(min=1e-4, max=1.0)
