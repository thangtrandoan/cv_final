from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

from models.detector import TinyGridDetector
from utils.box_ops import cxcywh_to_xyxy
from utils.json_utils import write_json
from utils.loss import decode_raw_predictions
from utils.nms import nms


TTA_BRIGHTNESS_FACTORS = [0.85, 1.15]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyGridDetector inference.")
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("./models/best.pth"))
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--conf_threshold", type=float, default=0.45)
    parser.add_argument("--nms_threshold", type=float, default=0.50)
    parser.add_argument("--max_detections_per_image", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def image_to_tensor(image: Image.Image, img_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(img_size, img_size, 3)
    return data.permute(2, 0, 1).float().div(255.0).unsqueeze(0)


@torch.no_grad()
def predict_image(
    model: TinyGridDetector,
    image: Image.Image,
    image_id: str,
    class_names: list[str],
    anchors: torch.Tensor,
    img_size: int,
    conf_threshold: float,
    nms_threshold: float,
    device: torch.device,
) -> dict[str, object]:
    original_w, original_h = image.size
    tensor = image_to_tensor(image, img_size).to(device)
    raw = model(tensor)
    decoded = decode_raw_predictions(raw, anchors.to(device))[0]

    boxes_cxcywh = decoded[..., 0:4].reshape(-1, 4)
    object_scores = torch.sigmoid(decoded[..., 4].reshape(-1))
    class_probs = F.softmax(decoded[..., 5:].reshape(-1, len(class_names)), dim=1)
    class_scores, class_ids = class_probs.max(dim=1)
    scores = object_scores * class_scores

    keep = scores >= conf_threshold
    boxes_cxcywh = boxes_cxcywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if boxes_cxcywh.numel() == 0:
        return {"image_id": image_id, "boxes": []}

    boxes = cxcywh_to_xyxy(boxes_cxcywh)
    scale = torch.tensor([original_w, original_h, original_w, original_h], device=device, dtype=torch.float32)
    boxes = (boxes * scale).clamp(min=0)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(max=original_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(max=original_h)

    output_boxes: list[dict[str, object]] = []
    for class_id in class_ids.unique():
        class_mask = class_ids == class_id
        selected = nms(boxes[class_mask], scores[class_mask], nms_threshold)
        class_boxes = boxes[class_mask][selected]
        class_scores_selected = scores[class_mask][selected]
        for box, score in zip(class_boxes, class_scores_selected):
            xmin, ymin, xmax, ymax = box.tolist()
            if xmax <= xmin or ymax <= ymin:
                continue
            output_boxes.append(
                {
                    "class": class_names[int(class_id.item())],
                    "confidence": round(float(score.item()), 6),
                    "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                }
            )

    output_boxes.sort(key=lambda item: item["confidence"], reverse=True)
    return {"image_id": image_id, "boxes": output_boxes}


def merge_boxes(
    image_id: str,
    boxes: list[dict[str, object]],
    class_names: list[str],
    nms_threshold: float,
    max_detections_per_image: int,
    device: torch.device,
) -> dict[str, object]:
    if not boxes:
        return {"image_id": image_id, "boxes": []}

    merged: list[dict[str, object]] = []
    for class_name in class_names:
        class_boxes = [box for box in boxes if box["class"] == class_name]
        if not class_boxes:
            continue
        box_tensor = torch.tensor([box["bbox"] for box in class_boxes], dtype=torch.float32, device=device)
        score_tensor = torch.tensor([box["confidence"] for box in class_boxes], dtype=torch.float32, device=device)
        selected = nms(box_tensor, score_tensor, nms_threshold)
        for index in selected.tolist():
            merged.append(class_boxes[index])

    merged.sort(key=lambda item: item["confidence"], reverse=True)
    return {"image_id": image_id, "boxes": merged[:max_detections_per_image]}


def predict_with_tta(
    model: TinyGridDetector,
    image_path: Path,
    class_names: list[str],
    anchors: torch.Tensor,
    img_size: int,
    conf_threshold: float,
    nms_threshold: float,
    max_detections_per_image: int,
    device: torch.device,
) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    original_w, _ = image.size
    all_boxes: list[dict[str, object]] = []

    variants: list[tuple[Image.Image, bool]] = [(image, False)]
    variants.append((image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), True))
    for factor in TTA_BRIGHTNESS_FACTORS:
        variants.append((ImageEnhance.Brightness(image).enhance(factor), False))

    for variant, flipped in variants:
        prediction = predict_image(
            model,
            variant,
            image_path.name,
            class_names,
            anchors,
            img_size,
            conf_threshold,
            nms_threshold,
            device,
        )
        for box in prediction["boxes"]:
            box = dict(box)
            if flipped:
                xmin, ymin, xmax, ymax = box["bbox"]
                box["bbox"] = [round(original_w - xmax, 2), ymin, round(original_w - xmin, 2), ymax]
            all_boxes.append(box)

    return merge_boxes(
        image_path.name,
        all_boxes,
        class_names,
        nms_threshold,
        max_detections_per_image,
        device,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(
        "Starting prediction "
        f"device={device} "
        f"image_dir={args.image_dir} "
        f"checkpoint={args.checkpoint} "
        f"conf_threshold={args.conf_threshold} "
        f"max_detections_per_image={args.max_detections_per_image} "
        f"tta=True "
        f"output={args.output}",
        flush=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    anchors = torch.tensor(checkpoint["anchors"], dtype=torch.float32, device=device)
    img_size = int(checkpoint.get("img_size", args.img_size))

    model = TinyGridDetector(
        num_classes=len(class_names),
        num_anchors=anchors.shape[0],
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_paths = sorted(
        [
            path
            for path in args.image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    predictions = [
        predict_with_tta(
            model,
            image_path,
            class_names,
            anchors,
            img_size,
            args.conf_threshold,
            args.nms_threshold,
            args.max_detections_per_image,
            device,
        )
        for image_path in image_paths
    ]
    write_json(predictions, args.output)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
