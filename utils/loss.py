from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .anchors import anchor_iou_wh


def encode_targets(
    targets: list[list[dict[str, Any]]],
    anchors: torch.Tensor,
    img_size: int,
    grid_size: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(targets)
    num_anchors = anchors.shape[0]
    encoded = torch.zeros((batch_size, grid_size, grid_size, num_anchors, 6), device=device)
    anchors = anchors.to(device)

    for batch_idx, image_targets in enumerate(targets):
        for item in image_targets:
            xmin, ymin, xmax, ymax = [float(value) for value in item["bbox"]]
            cx = ((xmin + xmax) / 2.0) / img_size
            cy = ((ymin + ymax) / 2.0) / img_size
            bw = max(1e-6, (xmax - xmin) / img_size)
            bh = max(1e-6, (ymax - ymin) / img_size)
            if bw <= 0 or bh <= 0:
                continue

            grid_x = min(grid_size - 1, max(0, int(cx * grid_size)))
            grid_y = min(grid_size - 1, max(0, int(cy * grid_size)))
            box_wh = torch.tensor([[bw, bh]], dtype=torch.float32, device=device)
            anchor_idx = int(anchor_iou_wh(box_wh, anchors).squeeze(0).argmax().item())

            encoded[batch_idx, grid_y, grid_x, anchor_idx, 0:4] = torch.tensor(
                [cx, cy, bw, bh], dtype=torch.float32, device=device
            )
            encoded[batch_idx, grid_y, grid_x, anchor_idx, 4] = 1.0
            encoded[batch_idx, grid_y, grid_x, anchor_idx, 5] = int(item["class_id"])

    return encoded


def decode_raw_predictions(raw: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    batch_size, _, grid_size, _ = raw.shape
    num_anchors = anchors.shape[0]
    num_classes = raw.shape[1] // num_anchors - 5
    pred = raw.view(batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
    pred = pred.permute(0, 3, 4, 1, 2).contiguous()

    device = raw.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij",
    )
    grid_x = grid_x.view(1, grid_size, grid_size, 1)
    grid_y = grid_y.view(1, grid_size, grid_size, 1)
    anchors = anchors.to(device).view(1, 1, 1, num_anchors, 2)

    xy = torch.empty_like(pred[..., 0:2])
    xy[..., 0] = (torch.sigmoid(pred[..., 0]) + grid_x) / grid_size
    xy[..., 1] = (torch.sigmoid(pred[..., 1]) + grid_y) / grid_size
    wh = anchors * torch.exp(pred[..., 2:4]).clamp(max=4.0)
    return torch.cat((xy, wh, pred[..., 4:]), dim=-1)


class DetectionLoss(nn.Module):
    def __init__(
        self,
        anchors: torch.Tensor,
        img_size: int = 416,
        grid_size: int = 13,
        num_classes: int = 5,
        lambda_box: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.3,
        lambda_cls: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("anchors", anchors.float())
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.register_buffer("class_weights", class_weights.float() if class_weights is not None else torch.ones(num_classes))

    def forward(self, raw: torch.Tensor, targets: list[list[dict[str, Any]]]) -> tuple[torch.Tensor, dict[str, float]]:
        target = encode_targets(
            targets,
            self.anchors,
            self.img_size,
            self.grid_size,
            self.num_classes,
            raw.device,
        )
        pred = decode_raw_predictions(raw, self.anchors)

        object_mask = target[..., 4] == 1
        no_object_mask = ~object_mask

        if object_mask.any():
            box_loss = F.smooth_l1_loss(pred[..., 0:4][object_mask], target[..., 0:4][object_mask], reduction="mean")
            class_logits = pred[..., 5:][object_mask]
            class_targets = target[..., 5][object_mask].long()
            cls_loss = F.cross_entropy(class_logits, class_targets, weight=self.class_weights.to(raw.device))
        else:
            box_loss = raw.sum() * 0.0
            cls_loss = raw.sum() * 0.0

        obj_logits = pred[..., 4]
        obj_loss = F.binary_cross_entropy_with_logits(obj_logits[object_mask], target[..., 4][object_mask], reduction="mean") if object_mask.any() else raw.sum() * 0.0
        noobj_loss = F.binary_cross_entropy_with_logits(
            obj_logits[no_object_mask], target[..., 4][no_object_mask], reduction="mean"
        )

        loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_cls * cls_loss
        )
        metrics = {
            "loss": float(loss.detach().cpu()),
            "box_loss": float(box_loss.detach().cpu()),
            "obj_loss": float(obj_loss.detach().cpu()),
            "noobj_loss": float(noobj_loss.detach().cpu()),
            "cls_loss": float(cls_loss.detach().cpu()),
        }
        return loss, metrics
